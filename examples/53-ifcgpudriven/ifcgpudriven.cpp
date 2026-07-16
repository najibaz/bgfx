/*
 * IFC GPU-driven viewer.
 *
 * Loads instanced geometry from BimInspector IFC->SQLite databases and renders
 * it with a fully GPU-driven pipeline:
 *   - all geometry in one master vertex/index buffer
 *   - per-instance {transform, color, draw params, world AABB}
 *   - a compute pass culls (frustum) and appends DrawIndexedIndirect commands
 *     via an atomic counter (compaction)
 *   - a single indirect draw with a GPU-provided draw count renders the visible set
 *
 * Usage: 53-ifcgpudriven <file1.db> [file2.db ...]
 */

#include "common.h"
#include "bgfx_utils.h"
#include "imgui/imgui.h"

#include <bx/allocator.h>
#include <bx/math.h>
#include <bx/file.h>
#include <bx/string.h>
#include <cstdlib>

#include "sqlite3.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
# define WIN32_LEAN_AND_MEAN
# include <windows.h>
#endif

namespace
{

// ---------------------------------------------------------------------------
// data layouts

struct PosVertex  // quantized 16-bit local position; dequant folded into the per-instance matrix
{
	static void init()
	{
		ms_layout.begin()
			.add(bgfx::Attrib::Position, 4, bgfx::AttribType::Int16, true /*normalized*/)  // 4th = pad -> 8-byte stride
			.end();
	}
	static bgfx::VertexLayout ms_layout;
};
bgfx::VertexLayout PosVertex::ms_layout;

// per-instance render data streamed as bgfx instance data (i_data0..i_data4)
struct InstanceRender
{
	float mtx[16];  // model matrix columns -> i_data0..i_data3
	float color[4]; // i_data4
};

// ---------------------------------------------------------------------------
// helpers

sqlite3* openDb(const std::string& path)
{
	std::string p = path;
#ifdef _WIN32
	// argv arrives in the ANSI code page; sqlite wants UTF-8
	int wlen = MultiByteToWideChar(CP_ACP, 0, path.c_str(), int(path.size()), nullptr, 0);
	if (wlen > 0)
	{
		std::wstring w(wlen, L'\0');
		MultiByteToWideChar(CP_ACP, 0, path.c_str(), int(path.size()), &w[0], wlen);
		int ulen = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), wlen, nullptr, 0, nullptr, nullptr);
		if (ulen > 0)
		{
			p.assign(ulen, '\0');
			WideCharToMultiByte(CP_UTF8, 0, w.c_str(), wlen, &p[0], ulen, nullptr, nullptr);
		}
	}
#endif
	sqlite3* db = nullptr;
	if (sqlite3_open_v2(p.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK)
	{
		if (db) sqlite3_close(db);
		return nullptr;
	}
	return db;
}

// quaternion (x,y,z,w) -> column-major 4x4 with scale and translation.
// Produces M such that mtxFromCols(cols)*v == T*R*S*v.
void trsToMatrix(const float* trs, float* m)
{
	const float sx = trs[0], sy = trs[1], sz = trs[2];
	const float tx = trs[3], ty = trs[4], tz = trs[5];
	const float qx = trs[6], qy = trs[7], qz = trs[8], qw = trs[9];

	// rotation matrix (row-major Rij)
	const float R00 = 1.0f - 2.0f * (qy*qy + qz*qz);
	const float R01 = 2.0f * (qx*qy - qw*qz);
	const float R02 = 2.0f * (qx*qz + qw*qy);
	const float R10 = 2.0f * (qx*qy + qw*qz);
	const float R11 = 1.0f - 2.0f * (qx*qx + qz*qz);
	const float R12 = 2.0f * (qy*qz - qw*qx);
	const float R20 = 2.0f * (qx*qz - qw*qy);
	const float R21 = 2.0f * (qy*qz + qw*qx);
	const float R22 = 1.0f - 2.0f * (qx*qx + qy*qy);

	// column-major storage: col0 = m[0..3], col1 = m[4..7], ...
	m[0]  = R00 * sx; m[1]  = R10 * sx; m[2]  = R20 * sx; m[3]  = 0.0f;
	m[4]  = R01 * sy; m[5]  = R11 * sy; m[6]  = R21 * sy; m[7]  = 0.0f;
	m[8]  = R02 * sz; m[9]  = R12 * sz; m[10] = R22 * sz; m[11] = 0.0f;
	m[12] = tx;       m[13] = ty;       m[14] = tz;       m[15] = 1.0f;
}

// ---------------------------------------------------------------------------
// scene held on the CPU before upload

struct Scene
{
	std::vector<int16_t>  vertices;   // quantized xyz per vertex (master buffer)
	std::vector<uint32_t> indices;    // master index buffer

	std::vector<InstanceRender> instances;   // render stream
	std::vector<uint32_t>       drawParams;  // uvec4 per instance: indexCount, firstIndex, baseVertex, 0
	std::vector<float>          bbox;         // 2 vec4 per instance: min.xyz,0 max.xyz,0

	float center[3] = { 0, 0, 0 };
	float radius = 1.0f;
	size_t numTriangles = 0;
	// quantization diagnostics
	float maxQuantErr = 0.0f;   // worst |original - dequantized| (world units)
	float worstExtent = 0.0f;   // that geometry's AABB extent on the worst axis
};

void loadDatabase(const std::string& path, Scene& scene, float bmin[3], float bmax[3])
{
	sqlite3* db = openDb(path);
	if (!db)
	{
		bx::debugPrintf("Cannot open database: %s\n", path.c_str());
		return;
	}
	float dbmin[3] = {  std::numeric_limits<float>::max(),  std::numeric_limits<float>::max(),  std::numeric_limits<float>::max() };
	float dbmax[3] = { -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max() };

	// 1) geometries -> owned CPU buffers keyed by Instance
	struct Geom
	{
		const void* vBlob; int vBytes;
		const void* iBlob; int iBytes;
		float color[3];
		float lmin[3], lmax[3];
	};
	// we must copy blob data out before stepping to the next row, so store copies
	struct Segment
	{
		uint32_t offset;      // first index within the geometry's index buffer
		uint32_t indexCount;  // number of indices (triangles * 3)
		float color[3];
		float alpha;
	};
	struct OwnedGeom
	{
		std::vector<float>    verts;
		std::vector<uint32_t> idx;
		std::vector<Segment>  segs;  // submeshes (material ranges)
		float lmin[3], lmax[3];
	};
	std::unordered_map<int64_t, OwnedGeom> geomMap;

	{
		sqlite3_stmt* st = nullptr;
		sqlite3_prepare_v2(db,
			"SELECT Instance, Vertices, FacesIndices, Materials FROM IFC_Item_Geometry "
			"WHERE Vertices IS NOT NULL AND FacesIndices IS NOT NULL",
			-1, &st, nullptr);
		while (st && sqlite3_step(st) == SQLITE_ROW)
		{
			int64_t inst = sqlite3_column_int64(st, 0);
			const float* v = (const float*)sqlite3_column_blob(st, 1);
			int vBytes = sqlite3_column_bytes(st, 1);
			const uint32_t* ix = (const uint32_t*)sqlite3_column_blob(st, 2);
			int iBytes = sqlite3_column_bytes(st, 2);
			const float* mat = (const float*)sqlite3_column_blob(st, 3);
			int mBytes = sqlite3_column_bytes(st, 3);

			uint32_t nv = uint32_t(vBytes / (3 * sizeof(float)));
			uint32_t ni = uint32_t(iBytes / sizeof(uint32_t));
			if (nv == 0 || ni == 0)
				continue;

			OwnedGeom g;
			g.verts.assign(v, v + nv * 3);
			g.idx.assign(ix, ix + ni);

			// parse Materials into submesh segments: [offset(indices), count(triangles), R,G,B,A]
			uint32_t numMatFloats = uint32_t(mBytes / sizeof(float));
			for (uint32_t s = 0; s + 6 <= numMatFloats; s += 6)
			{
				Segment seg;
				seg.offset     = uint32_t(mat[s + 0]);
				seg.indexCount = uint32_t(mat[s + 1]) * 3u;
				seg.color[0]   = mat[s + 2];
				seg.color[1]   = mat[s + 3];
				seg.color[2]   = mat[s + 4];
				seg.alpha      = mat[s + 5];
				if (seg.indexCount > 0 && seg.offset + seg.indexCount <= ni)
					g.segs.push_back(seg);
			}
			// fallback: whole geometry as one gray opaque submesh
			if (g.segs.empty())
			{
				Segment seg;
				seg.offset = 0; seg.indexCount = ni;
				seg.color[0] = 0.72f; seg.color[1] = 0.72f; seg.color[2] = 0.74f;
				seg.alpha = 1.0f;
				g.segs.push_back(seg);
			}
			g.lmin[0] = g.lmin[1] = g.lmin[2] =  std::numeric_limits<float>::max();
			g.lmax[0] = g.lmax[1] = g.lmax[2] = -std::numeric_limits<float>::max();
			for (uint32_t k = 0; k < nv; k++)
			{
				for (int c = 0; c < 3; c++)
				{
					float p = g.verts[k * 3 + c];
					g.lmin[c] = bx::min(g.lmin[c], p);
					g.lmax[c] = bx::max(g.lmax[c], p);
				}
			}
			geomMap.emplace(inst, std::move(g));
		}
		sqlite3_finalize(st);
	}

	// large structural element types used as Hi-Z occluders
	auto isOccluderType = [](const char* t) -> bool
	{
		if (!t) return false;
		static const char* kOccluders[] = {
			"IfcWall", "IfcWallStandardCase", "IfcSlab", "IfcColumn", "IfcRoof",
			"IfcCovering", "IfcBeam", "IfcCurtainWall", "IfcPlate", "IfcMember",
			"IfcStair", "IfcStairFlight", "IfcRamp", "IfcRampFlight", "IfcFooting",
		};
		for (const char* o : kOccluders)
			if (strcmp(t, o) == 0) return true;
		return false;
	};

	// 2) items -> group transforms by GeometryInstance (skip spaces + openings)
	struct Placement { std::vector<float> trs; bool occluder; };
	std::unordered_map<int64_t, std::vector<Placement>> instanceMap;
	{
		sqlite3_stmt* st = nullptr;
		sqlite3_prepare_v2(db,
			"SELECT GeometryInstance, TransformationMatrix, IfcType FROM IFC_Item "
			"WHERE GeometryInstance IS NOT NULL "
			"AND IfcType <> 'IfcSpace' AND IfcType <> 'IfcOpeningElement'",
			-1, &st, nullptr);
		while (st && sqlite3_step(st) == SQLITE_ROW)
		{
			int64_t g = sqlite3_column_int64(st, 0);
			if (geomMap.find(g) == geomMap.end())
				continue;
			const float* t = (const float*)sqlite3_column_blob(st, 1);
			int tBytes = sqlite3_column_bytes(st, 1);
			const char* type = (const char*)sqlite3_column_text(st, 2);
			std::vector<float> trs(10, 0.0f);
			trs[0] = trs[1] = trs[2] = 1.0f; // default scale
			trs[9] = 1.0f;                   // default quat w
			if (t && tBytes >= int(10 * sizeof(float)))
				memcpy(trs.data(), t, 10 * sizeof(float));
			instanceMap[g].push_back({ std::move(trs), isOccluderType(type) });
		}
		sqlite3_finalize(st);
	}

	sqlite3_close(db);

	// 3) build master buffers + per-instance data
	for (auto& kv : instanceMap)
	{
		auto it = geomMap.find(kv.first);
		if (it == geomMap.end())
			continue;
		OwnedGeom& g = it->second;

		uint32_t baseVertex     = uint32_t(scene.vertices.size() / 4);
		uint32_t geomBaseIndex  = uint32_t(scene.indices.size());

		// signed-normalized int16 quantization over the geometry's local AABB:
		//   a = (p - center) / half  in [-1,1];  stored int16 = round(a * 32767)
		//   bgfx SNORM dequant gives a back; local = center + half * a  (folded into M')
		float half[3], center[3];
		for (int c = 0; c < 3; c++)
		{
			half[c]   = 0.5f * (g.lmax[c] - g.lmin[c]);
			center[c] = 0.5f * (g.lmax[c] + g.lmin[c]);
		}

		// append geometry once (shared via baseVertex); positions -> normalized int16
		for (size_t k = 0; k < g.verts.size(); k += 3)
		{
			for (int c = 0; c < 3; c++)
			{
				float a = half[c] > 0.0f ? (g.verts[k + c] - center[c]) / half[c] : 0.0f;
				a = a < -1.0f ? -1.0f : (a > 1.0f ? 1.0f : a);
				int16_t iv = int16_t(a * 32767.0f + (a >= 0.0f ? 0.5f : -0.5f));
				scene.vertices.push_back(iv);

				// diagnostic: dequantize and compare to the original position
				float deq = center[c] + half[c] * (float(iv) / 32767.0f);
				float err = deq - g.verts[k + c];
				if (err < 0.0f) err = -err;
				if (err > scene.maxQuantErr)
				{
					scene.maxQuantErr = err;
					scene.worstExtent = 2.0f * half[c];
				}
			}
			scene.vertices.push_back(int16_t(0));  // w pad -> 8-byte vertex stride
		}
		scene.indices.insert(scene.indices.end(), g.idx.begin(), g.idx.end());

		for (auto& pl : kv.second)
		{
			float mtx[16];
			trsToMatrix(pl.trs.data(), mtx);

			// world AABB from local corners (once per instance)
			float wmin[3] = {  std::numeric_limits<float>::max(),  std::numeric_limits<float>::max(),  std::numeric_limits<float>::max() };
			float wmax[3] = { -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max() };
			for (int cx = 0; cx < 2; cx++)
			for (int cy = 0; cy < 2; cy++)
			for (int cz = 0; cz < 2; cz++)
			{
				float lx = cx ? g.lmax[0] : g.lmin[0];
				float ly = cy ? g.lmax[1] : g.lmin[1];
				float lz = cz ? g.lmax[2] : g.lmin[2];
				float wx = mtx[0]*lx + mtx[4]*ly + mtx[8]*lz  + mtx[12];
				float wy = mtx[1]*lx + mtx[5]*ly + mtx[9]*lz  + mtx[13];
				float wz = mtx[2]*lx + mtx[6]*ly + mtx[10]*lz + mtx[14];
				wmin[0] = bx::min(wmin[0], wx); wmax[0] = bx::max(wmax[0], wx);
				wmin[1] = bx::min(wmin[1], wy); wmax[1] = bx::max(wmax[1], wy);
				wmin[2] = bx::min(wmin[2], wz); wmax[2] = bx::max(wmax[2], wz);
			}
			// skip outlier geometry: stray verts / mis-georeferenced elements
			// (coords beyond +/-50 km) otherwise blow up the scene bounds & camera
			bool outlier = false;
			for (int c = 0; c < 3; c++)
				if (bx::abs(wmin[c]) > 50000.0f || bx::abs(wmax[c]) > 50000.0f)
					outlier = true;
			if (outlier)
				continue;

			for (int c = 0; c < 3; c++)
			{
				bmin[c] = bx::min(bmin[c], wmin[c]);
				bmax[c] = bx::max(bmax[c], wmax[c]);
				dbmin[c] = bx::min(dbmin[c], wmin[c]);
				dbmax[c] = bx::max(dbmax[c], wmax[c]);
			}

			// fold position dequant into the model matrix: M' = M * Squant(qs, qoff)
			// so the vertex shader (unchanged) maps stored int16 q -> world via M'.
			float mp[16];
			for (int i = 0; i < 4; i++)
			{
				mp[0 + i]  = mtx[0 + i] * half[0];
				mp[4 + i]  = mtx[4 + i] * half[1];
				mp[8 + i]  = mtx[8 + i] * half[2];
				mp[12 + i] = mtx[0 + i]*center[0] + mtx[4 + i]*center[1] + mtx[8 + i]*center[2] + mtx[12 + i];
			}

			// one draw item per submesh (material segment), classified independently
			for (const Segment& seg : g.segs)
			{
				InstanceRender ir;
				memcpy(ir.mtx, mp, sizeof(mp));
				ir.color[0] = seg.color[0]; ir.color[1] = seg.color[1];
				ir.color[2] = seg.color[2]; ir.color[3] = seg.alpha;
				scene.instances.push_back(ir);

				const uint32_t transparent = (seg.alpha < 0.99f) ? 1u : 0u;
				const uint32_t occluder    = pl.occluder ? 2u : 0u;  // bit1
				scene.drawParams.push_back(seg.indexCount);
				scene.drawParams.push_back(geomBaseIndex + seg.offset);
				scene.drawParams.push_back(baseVertex);
				scene.drawParams.push_back(transparent | occluder);

				scene.bbox.push_back(wmin[0]); scene.bbox.push_back(wmin[1]); scene.bbox.push_back(wmin[2]); scene.bbox.push_back(0.0f);
				scene.bbox.push_back(wmax[0]); scene.bbox.push_back(wmax[1]); scene.bbox.push_back(wmax[2]); scene.bbox.push_back(0.0f);

				scene.numTriangles += seg.indexCount / 3;
			}
		}
	}

	// per-DB bounds (to spot georeferenced / mis-placed models)
	if (dbmin[0] <= dbmax[0])
	{
		bx::debugPrintf("   bounds center (%.0f, %.0f, %.0f), extent %.0f m\n",
			0.5f*(dbmin[0]+dbmax[0]), 0.5f*(dbmin[1]+dbmax[1]), 0.5f*(dbmin[2]+dbmax[2]),
			bx::max(bx::max(dbmax[0]-dbmin[0], dbmax[1]-dbmin[1]), dbmax[2]-dbmin[2]));
	}
}

// Pre-count totals across all databases and reserve the scene vectors up front,
// so accumulation doesn't repeatedly reallocate the large master buffers.
void reserveScene(const std::vector<std::string>& dbPaths, Scene& scene)
{
	size_t totalVertFloats = 0, totalIndices = 0, totalItems = 0;
	for (const std::string& path : dbPaths)
	{
		sqlite3* db = openDb(path);
		if (!db) continue;
		sqlite3_stmt* st = nullptr;
		if (sqlite3_prepare_v2(db,
			"SELECT COALESCE(SUM(length(Vertices)),0), COALESCE(SUM(length(FacesIndices)),0) "
			"FROM IFC_Item_Geometry WHERE Vertices IS NOT NULL AND FacesIndices IS NOT NULL",
			-1, &st, nullptr) == SQLITE_OK && sqlite3_step(st) == SQLITE_ROW)
		{
			totalVertFloats += size_t(sqlite3_column_int64(st, 0)) / sizeof(float);
			totalIndices    += size_t(sqlite3_column_int64(st, 1)) / sizeof(uint32_t);
		}
		sqlite3_finalize(st);
		st = nullptr;
		if (sqlite3_prepare_v2(db,
			"SELECT COUNT(*) FROM IFC_Item WHERE GeometryInstance IS NOT NULL "
			"AND IfcType<>'IfcSpace' AND IfcType<>'IfcOpeningElement'",
			-1, &st, nullptr) == SQLITE_OK && sqlite3_step(st) == SQLITE_ROW)
		{
			totalItems += size_t(sqlite3_column_int64(st, 0));
		}
		sqlite3_finalize(st);
		sqlite3_close(db);
	}
	scene.vertices.reserve(totalVertFloats / 3 * 4);  // 4 int16 per vertex (xyz + pad)
	scene.indices.reserve(totalIndices);
	const size_t est = totalItems * 2;  // draw-items ~= items * avg submeshes
	scene.instances.reserve(est);
	scene.drawParams.reserve(est * 4);
	scene.bbox.reserve(est * 8);
}

// ---------------------------------------------------------------------------

class IfcGpuDriven : public entry::AppI
{
public:
	IfcGpuDriven(const char* _name, const char* _description, const char* _url)
		: entry::AppI(_name, _description, _url)
	{
	}

	void init(int32_t _argc, const char* const* _argv, uint32_t _width, uint32_t _height) override
	{
		Args args(_argc, _argv);

		m_width  = _width;
		m_height = _height;
		m_debug  = BGFX_DEBUG_TEXT;
		m_reset  = BGFX_RESET_NONE;  // no vsync -> uncapped fps

		bgfx::Init init;
		init.type     = bgfx::RendererType::Vulkan;  // need DRAW_INDIRECT_COUNT
		init.vendorId = args.m_pciId;
		init.platformData.nwh  = entry::getNativeWindowHandle(entry::kDefaultWindowHandle);
		init.platformData.ndt  = entry::getNativeDisplayHandle();
		init.platformData.type = entry::getNativeWindowHandleType();
		init.resolution.width  = m_width;
		init.resolution.height = m_height;
		init.resolution.reset  = m_reset;
		bgfx::init(init);

		bgfx::setDebug(m_debug);

		PosVertex::init();

		// collect .db paths from the command line
		std::vector<std::string> dbPaths;
		for (int i = 1; i < _argc; i++)
		{
			const char* a = _argv[i];
			size_t len = strlen(a);
			if (len >= 3 && strcmp(a + len - 3, ".db") == 0)
				dbPaths.push_back(a);
		}

		// load scene
		float bmin[3] = {  std::numeric_limits<float>::max(),  std::numeric_limits<float>::max(),  std::numeric_limits<float>::max() };
		float bmax[3] = { -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max() };
		const int64_t tLoad = bx::getHPCounter();
		reserveScene(dbPaths, m_scene);
		for (const std::string& p : dbPaths)
		{
			bx::debugPrintf("Loading %s ...\n", p.c_str());
			loadDatabase(p, m_scene, bmin, bmax);
		}
		bx::debugPrintf("Load time: %.2f s\n",
			double(bx::getHPCounter() - tLoad) / double(bx::getHPFrequency()));

		m_numInstances = uint32_t(m_scene.instances.size());
		if (m_numInstances == 0)
		{
			m_scene.center[0] = m_scene.center[1] = m_scene.center[2] = 0.0f;
			m_scene.radius = 1.0f;
		}
		else
		{
			for (int c = 0; c < 3; c++)
				m_scene.center[c] = 0.5f * (bmin[c] + bmax[c]);
			float dx = bmax[0]-bmin[0], dy = bmax[1]-bmin[1], dz = bmax[2]-bmin[2];
			m_scene.radius = 0.5f * bx::sqrt(dx*dx + dy*dy + dz*dz);
			if (m_scene.radius < 1e-3f) m_scene.radius = 1.0f;
		}
		bx::debugPrintf("Buffers: vtx %.0f MB, idx %.0f MB, instance %.0f MB\n",
			double(m_scene.vertices.size() * sizeof(int16_t)) / 1e6,
			double(m_scene.indices.size() * sizeof(uint32_t)) / 1e6,
			double(m_scene.instances.size() * sizeof(InstanceRender)) / 1e6);
		bx::debugPrintf("Quant error: max %.4f m (%.2f mm), worst-element extent %.2f m\n",
			m_scene.maxQuantErr, m_scene.maxQuantErr * 1000.0f, m_scene.worstExtent);
		bx::debugPrintf("Instances: %u, triangles: %llu, radius: %f\n",
			m_numInstances, (unsigned long long)m_scene.numTriangles, m_scene.radius);

		createBuffers();

		// programs
		m_progRender    = loadProgram("vs_ifc", "fs_ifc");
		m_progDepth     = bgfx::createProgram(loadShader("vs_ifc_depth"), true);  // vertex-only (depth target)
		m_progReset     = bgfx::createProgram(loadShader("cs_ifc_reset"), true);
		m_progCull      = bgfx::createProgram(loadShader("cs_ifc_cull"),  true);
		// reuse example 37's generic Hi-Z build shaders (already compiled to spirv)
		m_progCopyZ     = bgfx::createProgram(loadShader("cs_gdr_copy_z"),         true);
		m_progDownscale = bgfx::createProgram(loadShader("cs_gdr_downscale_hi_z"), true);

		u_cullParams        = bgfx::createUniform("u_cullParams",        bgfx::UniformType::Vec4);
		u_hiZSize           = bgfx::createUniform("u_hiZSize",           bgfx::UniformType::Vec4);
		u_screenPixels      = bgfx::createUniform("u_screenPixels",      bgfx::UniformType::Vec4);
		u_inputRTSize       = bgfx::createUniform("u_inputRTSize",       bgfx::UniformType::Vec4);
		s_hiZ               = bgfx::createUniform("s_hiZ",               bgfx::UniformType::Sampler);
		s_texOcclusionDepth = bgfx::createUniform("s_texOcclusionDepth", bgfx::UniformType::Sampler);

		// camera
		m_heading   = 0.7f;
		m_pitch     = 0.5f;
		m_distance  = m_scene.radius * 2.0f;
		m_target    = bx::Vec3(m_scene.center[0], m_scene.center[1], m_scene.center[2]);
		m_frustumCull = true;

		bgfx::setViewClear(kViewDepth, BGFX_CLEAR_DEPTH, 0x000000ff, 1.0f, 0);
		bgfx::setViewClear(kViewRender, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 0x303030ff, 1.0f, 0);

		const char* shot = getenv("IFC_SHOT");
		if (shot)
			bx::strCopy(m_shotPath, sizeof(m_shotPath), shot);

		imguiCreate();
		m_timeOffset = bx::getHPCounter();
	}

	void createBuffers()
	{
		if (m_numInstances == 0)
			return;

		m_vbh = bgfx::createVertexBuffer(
			bgfx::copy(m_scene.vertices.data(), uint32_t(m_scene.vertices.size() * sizeof(int16_t))),
			PosVertex::ms_layout);
		m_ibh = bgfx::createIndexBuffer(
			bgfx::copy(m_scene.indices.data(), uint32_t(m_scene.indices.size() * sizeof(uint32_t))),
			BGFX_BUFFER_INDEX32);

		// instance render stream: 5 x vec4 (mat4 + color)
		m_instanceLayout.begin()
			.add(bgfx::Attrib::TexCoord0, 4, bgfx::AttribType::Float)
			.add(bgfx::Attrib::TexCoord1, 4, bgfx::AttribType::Float)
			.add(bgfx::Attrib::TexCoord2, 4, bgfx::AttribType::Float)
			.add(bgfx::Attrib::TexCoord3, 4, bgfx::AttribType::Float)
			.add(bgfx::Attrib::TexCoord4, 4, bgfx::AttribType::Float)
			.end();
		m_instanceVbh = bgfx::createVertexBuffer(
			bgfx::copy(m_scene.instances.data(), uint32_t(m_scene.instances.size() * sizeof(InstanceRender))),
			m_instanceLayout);

		// per-instance draw params (uvec4) - compute read
		m_drawParamsBuf = bgfx::createIndexBuffer(
			bgfx::copy(m_scene.drawParams.data(), uint32_t(m_scene.drawParams.size() * sizeof(uint32_t))),
			BGFX_BUFFER_INDEX32 | BGFX_BUFFER_COMPUTE_READ);

		// per-instance world AABB (vec4 pairs) - compute read
		m_bboxLayout.begin().add(bgfx::Attrib::TexCoord0, 4, bgfx::AttribType::Float).end();
		m_bboxBuf = bgfx::createVertexBuffer(
			bgfx::copy(m_scene.bbox.data(), uint32_t(m_scene.bbox.size() * sizeof(float))),
			m_bboxLayout, BGFX_BUFFER_COMPUTE_READ);

		// ping-pong draw counters + indirect command buffers.
		// The counter must be a static IndexBufferHandle to serve as submit()'s
		// draw-count handle; compute flags make it GPU-writable.
		const uint32_t zero = 0;
		for (int i = 0; i < 2; i++)
		{
			m_count[i] = bgfx::createIndexBuffer(
				bgfx::copy(&zero, sizeof(uint32_t)),
				BGFX_BUFFER_INDEX32 | BGFX_BUFFER_COMPUTE_READ_WRITE | BGFX_BUFFER_DRAW_INDIRECT);
			m_indirect[i] = bgfx::createIndirectBuffer(m_numInstances);
		}
		m_transpCount = bgfx::createIndexBuffer(
			bgfx::copy(&zero, sizeof(uint32_t)),
			BGFX_BUFFER_INDEX32 | BGFX_BUFFER_COMPUTE_READ_WRITE | BGFX_BUFFER_DRAW_INDIRECT);
		m_transpIndirect = bgfx::createIndirectBuffer(m_numInstances);
		for (int i = 0; i < 2; i++)
		{
			m_occCount[i] = bgfx::createIndexBuffer(
				bgfx::copy(&zero, sizeof(uint32_t)),
				BGFX_BUFFER_INDEX32 | BGFX_BUFFER_COMPUTE_READ_WRITE | BGFX_BUFFER_DRAW_INDIRECT);
			m_occIndirect[i] = bgfx::createIndirectBuffer(m_numInstances);
		}

		// Hi-Z resources: pow2 depth target + mip pyramid
		m_hizW = 1u << uint32_t(bx::floor(bx::log2(float(m_width))));
		m_hizH = 1u << uint32_t(bx::floor(bx::log2(float(m_height))));
		m_hizW = bx::max<uint32_t>(m_hizW, 16);
		m_hizH = bx::max<uint32_t>(m_hizH, 16);
		m_hizMips = uint8_t(1 + bx::floor(bx::log2(float(bx::max(m_hizW, m_hizH)))));

		const uint64_t tsFlags = 0
			| BGFX_TEXTURE_RT
			| BGFX_SAMPLER_MIN_POINT | BGFX_SAMPLER_MAG_POINT | BGFX_SAMPLER_MIP_POINT
			| BGFX_SAMPLER_U_CLAMP | BGFX_SAMPLER_V_CLAMP;
		m_depthFb = bgfx::createFrameBuffer(uint16_t(m_hizW), uint16_t(m_hizH),
			bgfx::TextureFormat::D32F, tsFlags);
		m_hizTex = bgfx::createTexture2D(uint16_t(m_hizW), uint16_t(m_hizH), true, 1,
			bgfx::TextureFormat::R32F, BGFX_TEXTURE_COMPUTE_WRITE | tsFlags);
	}

	virtual int shutdown() override
	{
		imguiDestroy();
		if (bgfx::isValid(m_vbh))          bgfx::destroy(m_vbh);
		if (bgfx::isValid(m_ibh))          bgfx::destroy(m_ibh);
		if (bgfx::isValid(m_instanceVbh))  bgfx::destroy(m_instanceVbh);
		if (bgfx::isValid(m_drawParamsBuf))bgfx::destroy(m_drawParamsBuf);
		if (bgfx::isValid(m_bboxBuf))      bgfx::destroy(m_bboxBuf);
		for (int i = 0; i < 2; i++)
		{
			if (bgfx::isValid(m_count[i]))    bgfx::destroy(m_count[i]);
			if (bgfx::isValid(m_indirect[i])) bgfx::destroy(m_indirect[i]);
		}
		if (bgfx::isValid(m_transpCount))    bgfx::destroy(m_transpCount);
		if (bgfx::isValid(m_transpIndirect)) bgfx::destroy(m_transpIndirect);
		for (int i = 0; i < 2; i++)
		{
			if (bgfx::isValid(m_occCount[i]))    bgfx::destroy(m_occCount[i]);
			if (bgfx::isValid(m_occIndirect[i])) bgfx::destroy(m_occIndirect[i]);
		}
		if (bgfx::isValid(m_depthFb)) bgfx::destroy(m_depthFb);
		if (bgfx::isValid(m_hizTex))  bgfx::destroy(m_hizTex);
		bgfx::destroy(m_progRender);
		bgfx::destroy(m_progDepth);
		bgfx::destroy(m_progReset);
		bgfx::destroy(m_progCull);
		bgfx::destroy(m_progCopyZ);
		bgfx::destroy(m_progDownscale);
		bgfx::destroy(u_cullParams);
		bgfx::destroy(u_hiZSize);
		bgfx::destroy(u_screenPixels);
		bgfx::destroy(u_inputRTSize);
		bgfx::destroy(s_hiZ);
		bgfx::destroy(s_texOcclusionDepth);
		bgfx::shutdown();
		return 0;
	}

	bool update() override
	{
		if (entry::processEvents(m_width, m_height, m_debug, m_reset, &m_mouseState))
			return false;

		imguiBeginFrame(m_mouseState.m_mx, m_mouseState.m_my,
			(m_mouseState.m_buttons[entry::MouseButton::Left]   ? IMGUI_MBUT_LEFT   : 0) |
			(m_mouseState.m_buttons[entry::MouseButton::Right]  ? IMGUI_MBUT_RIGHT  : 0) |
			(m_mouseState.m_buttons[entry::MouseButton::Middle] ? IMGUI_MBUT_MIDDLE : 0),
			m_mouseState.m_mz, uint16_t(m_width), uint16_t(m_height));

		showExampleDialog(this);

		ImGui::SetNextWindowPos(ImVec2(m_width - m_width / 5.0f - 10.0f, 10.0f), ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(m_width / 5.0f, m_height / 5.0f), ImGuiCond_FirstUseEver);
		ImGui::Begin("Settings", NULL, 0);
		ImGui::Checkbox("Frustum cull", &m_frustumCull);
		ImGui::Checkbox("Occlusion cull (Hi-Z)", &m_occlusionCull);
		ImGui::SliderFloat("Min pixel size", &m_minPixelSize, 0.0f, 8.0f);
		ImGui::Text("Instances: %u", m_numInstances);
		ImGui::Text("Triangles: %llu", (unsigned long long)m_scene.numTriangles);
		ImGui::End();

		imguiEndFrame();

		// simple orbit camera
		updateCamera();

		float view[16];
		float proj[16];
		// Z-up scene: pass up = +Z. The default +Y up renders the model on its
		// side / upside-down.
		bx::mtxLookAt(view, m_eye, m_at, bx::Vec3(0.0f, 0.0f, 1.0f));
		bx::mtxProj(proj, 60.0f, float(m_width) / float(m_height), 0.05f,
			m_scene.radius * 6.0f + m_distance, bgfx::getCaps()->homogeneousDepth);

		if (m_numInstances > 0)
		{
			const int prev = m_pingpong;
			const int cur  = m_pingpong ^ 1;

			// pass 0: render previous-frame visible set to depth (temporal occluders)
			bgfx::setViewFrameBuffer(kViewDepth, m_depthFb);
			bgfx::setViewRect(kViewDepth, 0, 0, uint16_t(m_hizW), uint16_t(m_hizH));
			bgfx::setViewTransform(kViewDepth, view, proj);
			bgfx::setVertexBuffer(0, m_vbh);
			bgfx::setIndexBuffer(m_ibh);
			bgfx::setInstanceDataBuffer(m_instanceVbh, 0, m_numInstances);
			bgfx::setState(BGFX_STATE_WRITE_Z | BGFX_STATE_DEPTH_TEST_LESS);
			bgfx::submit(kViewDepth, m_progDepth, m_occIndirect[prev], 0, m_occCount[prev], 0, uint32_t(m_numInstances));

			// Hi-Z build: copy depth -> mip0
			{
				float rt[4] = { float(m_hizW), float(m_hizH), 0.0f, 0.0f };
				bgfx::setUniform(u_inputRTSize, rt);
				bgfx::setTexture(0, s_texOcclusionDepth, bgfx::getTexture(m_depthFb, 0));
				bgfx::setImage(1, m_hizTex, 0, bgfx::Access::Write, bgfx::TextureFormat::R32F);
				bgfx::dispatch(kViewHizCopy, m_progCopyZ, (m_hizW + 15) / 16, (m_hizH + 15) / 16, 1);
			}
			// Hi-Z build: downscale mips (max of 2x2)
			uint32_t mw = m_hizW, mh = m_hizH;
			for (uint8_t lod = 1; lod < m_hizMips; lod++)
			{
				mw = bx::max<uint32_t>(mw >> 1, 1);
				mh = bx::max<uint32_t>(mh >> 1, 1);
				float rt[4] = { float(mw), float(mh), 2.0f, 2.0f };
				bgfx::setUniform(u_inputRTSize, rt);
				bgfx::setImage(0, m_hizTex, lod - 1, bgfx::Access::Read,  bgfx::TextureFormat::R32F);
				bgfx::setImage(1, m_hizTex, lod,     bgfx::Access::Write, bgfx::TextureFormat::R32F);
				bgfx::dispatch(kViewHizDown, m_progDownscale, (mw + 15) / 16, (mh + 15) / 16, 1);
			}

			// pass: reset current draw counters (opaque + transparent)
			bgfx::setBuffer(0, m_count[cur], bgfx::Access::Write);
			bgfx::dispatch(kViewReset, m_progReset, 1, 1, 1);
			bgfx::setBuffer(0, m_transpCount, bgfx::Access::Write);
			bgfx::dispatch(kViewReset, m_progReset, 1, 1, 1);
			bgfx::setBuffer(0, m_occCount[cur], bgfx::Access::Write);
			bgfx::dispatch(kViewReset, m_progReset, 1, 1, 1);

			// pass: cull (frustum + Hi-Z occlusion) -> current commands
			bgfx::setViewTransform(kViewCull, view, proj);
			float cullParams[4] = {
				float(m_numInstances),
				m_frustumCull   ? 1.0f : 0.0f,
				m_occlusionCull ? 1.0f : 0.0f,
				float(m_hizMips - 1)
			};
			float hizSize[4] = { float(m_hizW), float(m_hizH), 1.0f / float(m_hizW), 1.0f / float(m_hizH) };
			bgfx::setUniform(u_cullParams, cullParams);
			bgfx::setUniform(u_hiZSize, hizSize);
			float screenPixels[4] = { float(m_width), float(m_height), m_minPixelSize, 0.0f };
			bgfx::setUniform(u_screenPixels, screenPixels);
			bgfx::setTexture(0, s_hiZ, m_hizTex);
			bgfx::setBuffer(1, m_drawParamsBuf,  bgfx::Access::Read);
			bgfx::setBuffer(2, m_bboxBuf,        bgfx::Access::Read);
			bgfx::setBuffer(3, m_count[cur],     bgfx::Access::ReadWrite);
			bgfx::setBuffer(4, m_indirect[cur],  bgfx::Access::Write);
			bgfx::setBuffer(5, m_transpCount,    bgfx::Access::ReadWrite);
			bgfx::setBuffer(6, m_transpIndirect, bgfx::Access::Write);
			bgfx::setBuffer(7, m_occCount[cur],  bgfx::Access::ReadWrite);
			bgfx::setBuffer(8, m_occIndirect[cur], bgfx::Access::Write);
			bgfx::dispatch(kViewCull, m_progCull, (m_numInstances + 63) / 64, 1, 1);

			// pass: render visible set via indirect draw with GPU draw count
			bgfx::setViewTransform(kViewRender, view, proj);
			bgfx::setViewRect(kViewRender, 0, 0, uint16_t(m_width), uint16_t(m_height));
			bgfx::setVertexBuffer(0, m_vbh);
			bgfx::setIndexBuffer(m_ibh);
			bgfx::setInstanceDataBuffer(m_instanceVbh, 0, m_numInstances);
			bgfx::setState(BGFX_STATE_DEFAULT);
			bgfx::submit(kViewRender, m_progRender, m_indirect[cur], 0, m_count[cur], 0, uint32_t(m_numInstances));

			// pass: render transparent visible set (alpha blend, depth test, no depth write)
			bgfx::setViewTransform(kViewRenderT, view, proj);
			bgfx::setViewRect(kViewRenderT, 0, 0, uint16_t(m_width), uint16_t(m_height));
			bgfx::setVertexBuffer(0, m_vbh);
			bgfx::setIndexBuffer(m_ibh);
			bgfx::setInstanceDataBuffer(m_instanceVbh, 0, m_numInstances);
			bgfx::setState(BGFX_STATE_WRITE_RGB | BGFX_STATE_DEPTH_TEST_LESS | BGFX_STATE_BLEND_ALPHA);
			bgfx::submit(kViewRenderT, m_progRender, m_transpIndirect, 0, m_transpCount, 0, uint32_t(m_numInstances));

			m_pingpong = cur;
		}
		else
		{
			bgfx::touch(kViewRender);
		}

		// one-shot screenshot for headless verification (set IFC_SHOT env var)
		if (m_shotPath[0] != '\0' && m_frameNo == 120)
			bgfx::requestScreenShot(BGFX_INVALID_HANDLE, m_shotPath);
		m_frameNo++;

		bgfx::frame();
		return true;
	}

	void updateCamera()
	{
		// Controls: left-drag orbit, right/middle-drag pan (moves target),
		// wheel zoom (multiplicative). Matches the bim-viewer camera.
		const bool overGui = ImGui::MouseOverArea();
		if (!overGui)
		{
			const float dx = float(m_mouseState.m_mx - m_prevMx);
			const float dy = float(m_mouseState.m_my - m_prevMy);
			if (m_mouseState.m_buttons[entry::MouseButton::Left])
			{
				if (m_dragging)
				{
					m_heading -= dx * 0.005f;
					m_pitch   += dy * 0.005f;
					m_pitch = bx::clamp(m_pitch, -1.5f, 1.5f);
				}
				m_dragging = true;
			}
			else if (m_mouseState.m_buttons[entry::MouseButton::Right]
			     ||  m_mouseState.m_buttons[entry::MouseButton::Middle])
			{
				if (m_dragging)
				{
					// Pan in the view plane, scaled with distance so the scene
					// tracks the cursor. Uses last frame's view basis.
					float v[16];
					bx::mtxLookAt(v, m_eye, m_at, bx::Vec3(0.0f, 0.0f, 1.0f));
					const bx::Vec3 right(v[0], v[4], v[8]);
					const bx::Vec3 up   (v[1], v[5], v[9]);
					const float s = m_distance * 0.0015f;
					m_target = bx::add(m_target,
						bx::add(bx::mul(right, -dx * s), bx::mul(up, dy * s)));
				}
				m_dragging = true;
			}
			else
			{
				m_dragging = false;
			}
			int mz = m_mouseState.m_mz - m_prevMz;
			if (mz != 0)
				m_distance *= bx::pow(0.9f, float(mz));
			m_distance = bx::clamp(m_distance,
				m_scene.radius * 0.001f, m_scene.radius * 20.0f);
		}
		m_prevMx = m_mouseState.m_mx;
		m_prevMy = m_mouseState.m_my;
		m_prevMz = m_mouseState.m_mz;

		// IFC is Z-up: orbit the eye around the movable target.
		float cp = bx::cos(m_pitch);
		float dir[3] = { cp * bx::sin(m_heading), cp * bx::cos(m_heading), bx::sin(m_pitch) };
		m_eye = bx::Vec3(
			m_target.x + dir[0] * m_distance,
			m_target.y + dir[1] * m_distance,
			m_target.z + dir[2] * m_distance);
		m_at = m_target;
	}

	// view ids (executed in ascending order)
	static const bgfx::ViewId kViewDepth   = 0;  // render previous-frame visible set to depth
	static const bgfx::ViewId kViewHizCopy = 1;  // depth -> Hi-Z mip 0
	static const bgfx::ViewId kViewHizDown = 2;  // downscale Hi-Z mips
	static const bgfx::ViewId kViewReset   = 3;  // zero draw counters
	static const bgfx::ViewId kViewCull    = 4;  // cull + append commands
	static const bgfx::ViewId kViewRender  = 5;  // render opaque visible set
	static const bgfx::ViewId kViewRenderT = 6;  // render transparent visible set (blended)

	entry::MouseState m_mouseState;
	uint32_t m_width = 0, m_height = 0, m_debug = 0, m_reset = 0;
	int64_t  m_timeOffset = 0;

	Scene m_scene;
	uint32_t m_numInstances = 0;

	bgfx::VertexLayout m_instanceLayout;
	bgfx::VertexLayout m_bboxLayout;
	bgfx::VertexBufferHandle       m_vbh = BGFX_INVALID_HANDLE;
	bgfx::IndexBufferHandle        m_ibh = BGFX_INVALID_HANDLE;
	bgfx::VertexBufferHandle       m_instanceVbh = BGFX_INVALID_HANDLE;
	bgfx::IndexBufferHandle        m_drawParamsBuf = BGFX_INVALID_HANDLE;
	bgfx::VertexBufferHandle       m_bboxBuf = BGFX_INVALID_HANDLE;
	// ping-pong indirect commands + draw counts (temporal occlusion)
	bgfx::IndexBufferHandle    m_count[2]    = { BGFX_INVALID_HANDLE, BGFX_INVALID_HANDLE };
	bgfx::IndirectBufferHandle m_indirect[2] = { BGFX_INVALID_HANDLE, BGFX_INVALID_HANDLE };
	int m_pingpong = 0;
	// transparent set (rendered after opaque; not used as occluders)
	bgfx::IndexBufferHandle    m_transpCount    = BGFX_INVALID_HANDLE;
	bgfx::IndirectBufferHandle m_transpIndirect = BGFX_INVALID_HANDLE;
	bgfx::IndexBufferHandle    m_occCount[2]    = { BGFX_INVALID_HANDLE, BGFX_INVALID_HANDLE };
	bgfx::IndirectBufferHandle m_occIndirect[2] = { BGFX_INVALID_HANDLE, BGFX_INVALID_HANDLE };

	// Hi-Z occlusion resources
	uint32_t m_hizW = 0, m_hizH = 0;
	uint8_t  m_hizMips = 0;
	bgfx::FrameBufferHandle m_depthFb = BGFX_INVALID_HANDLE;
	bgfx::TextureHandle     m_hizTex  = BGFX_INVALID_HANDLE;

	bgfx::ProgramHandle m_progRender    = BGFX_INVALID_HANDLE;
	bgfx::ProgramHandle m_progDepth     = BGFX_INVALID_HANDLE;
	bgfx::ProgramHandle m_progReset     = BGFX_INVALID_HANDLE;
	bgfx::ProgramHandle m_progCull      = BGFX_INVALID_HANDLE;
	bgfx::ProgramHandle m_progCopyZ     = BGFX_INVALID_HANDLE;
	bgfx::ProgramHandle m_progDownscale = BGFX_INVALID_HANDLE;
	bgfx::UniformHandle u_cullParams        = BGFX_INVALID_HANDLE;
	bgfx::UniformHandle u_hiZSize           = BGFX_INVALID_HANDLE;
	bgfx::UniformHandle u_screenPixels      = BGFX_INVALID_HANDLE;
	bgfx::UniformHandle u_inputRTSize       = BGFX_INVALID_HANDLE;
	bgfx::UniformHandle s_hiZ               = BGFX_INVALID_HANDLE;
	bgfx::UniformHandle s_texOcclusionDepth = BGFX_INVALID_HANDLE;
	bool m_occlusionCull = true;
	float m_minPixelSize = 2.0f;  // small-object cull threshold (projected px)

	// camera
	float m_heading = 0, m_pitch = 0, m_distance = 10.0f;
	bx::Vec3 m_target = bx::Vec3(0.0f);
	bx::Vec3 m_eye = bx::Vec3(0.0f);
	bx::Vec3 m_at  = bx::Vec3(0.0f);
	bool  m_dragging = false;
	int32_t m_prevMx = 0, m_prevMy = 0, m_prevMz = 0;
	bool  m_frustumCull = true;
	uint32_t m_frameNo = 0;
	char m_shotPath[512] = { 0 };
};

} // namespace

ENTRY_IMPLEMENT_MAIN(
	  IfcGpuDriven
	, "53-ifcgpudriven"
	, "IFC GPU-driven viewer (SQLite instanced geometry, compute cull + indirect draw)."
	, "https://bkaradzic.github.io/bgfx/examples.html"
	);

#ifndef GEOMETRY_GEOMETRY_CUH
#define GEOMETRY_GEOMETRY_CUH

#include <span>
#include <vector>

#include <cuda/buffer>
#include <cuda/memory_pool>
#include <cuda/stream>
#include <cuda_runtime.h>

namespace geometry {
    enum struct ShapeKind { sphere, box, cylinder, cone, torus, mesh };

    enum struct MotionKind { rigid, affine };

    struct Sphere {
        float radius{};

        bool operator==(const Sphere&) const = default;
    };

    struct Box {
        float3 half_extent{};

        bool operator==(const Box& other) const {
            return half_extent.x == other.half_extent.x && half_extent.y == other.half_extent.y && half_extent.z == other.half_extent.z;
        }
    };

    struct Cylinder {
        float radius{};
        float half_height{};

        bool operator==(const Cylinder&) const = default;
    };

    struct Cone {
        float lower_radius{};
        float upper_radius{};
        float half_height{};

        bool operator==(const Cone&) const = default;
    };

    struct Torus {
        float major_radius{};
        float minor_radius{};

        bool operator==(const Torus&) const = default;
    };

    struct Shape {
        ShapeKind kind{ShapeKind::sphere};
        union {
            Sphere sphere{};
            Box box;
            Cylinder cylinder;
            Cone cone;
            Torus torus;
            int mesh;
        };

        bool operator==(const Shape& other) const {
            if (kind != other.kind) return false;
            switch (kind) {
            case ShapeKind::sphere: return sphere == other.sphere;
            case ShapeKind::box: return box == other.box;
            case ShapeKind::cylinder: return cylinder == other.cylinder;
            case ShapeKind::cone: return cone == other.cone;
            case ShapeKind::torus: return torus == other.torus;
            case ShapeKind::mesh: return mesh == other.mesh;
            }
            return false;
        }
    };

    struct AffineMotion {
        float matrix[12]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
        float derivative[12]{};
        float time{};

        bool operator==(const AffineMotion&) const = default;
    };

    struct RigidMotion {
        float translation[3]{};
        float orientation[4]{0, 0, 0, 1};
        float linear_velocity[3]{};
        float angular_velocity[3]{};
        float time{};
        float scale{1};

        bool operator==(const RigidMotion&) const = default;
    };

    struct Motion {
        MotionKind kind{MotionKind::rigid};
        union {
            RigidMotion rigid{};
            AffineMotion affine;
        };

        bool operator==(const Motion& other) const {
            if (kind != other.kind) return false;
            return kind == MotionKind::rigid ? rigid == other.rigid : affine == other.affine;
        }
    };

    struct Instance {
        int shape{};
        Motion motion;

        bool operator==(const Instance&) const = default;
    };

    struct Bounds {
        float3 lower{};
        float3 upper{};

        bool operator==(const Bounds& other) const {
            return lower.x == other.lower.x && lower.y == other.lower.y && lower.z == other.lower.z && upper.x == other.upper.x && upper.y == other.upper.y && upper.z == other.upper.z;
        }
    };

    struct Transform {
        float matrix[12]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
        float3 inverse[3]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        float inverse_metric{1.0f / 3.0f};
    };

    struct BvhView {
        Bounds* nodes{};
        int* order{};
        int leaf_count{};

        bool operator==(const BvhView&) const = default;
    };

    struct MeshView {
        float3* positions{};
        int3* triangles{};
        int triangle_count{};
        Bounds bounds{};
        BvhView bvh;

        bool operator==(const MeshView&) const = default;
    };

    struct ShapeView {
        ShapeKind kind{ShapeKind::sphere};
        union {
            Sphere sphere{};
            Box box;
            Cylinder cylinder;
            Cone cone;
            Torus torus;
            MeshView mesh;
        };

        bool operator==(const ShapeView& other) const {
            if (kind != other.kind) return false;
            switch (kind) {
            case ShapeKind::sphere: return sphere == other.sphere;
            case ShapeKind::box: return box == other.box;
            case ShapeKind::cylinder: return cylinder == other.cylinder;
            case ShapeKind::cone: return cone == other.cone;
            case ShapeKind::torus: return torus == other.torus;
            case ShapeKind::mesh: return mesh == other.mesh;
            }
            return false;
        }
    };

    struct InstanceView {
        ShapeView shape;
        Motion motion;
        Bounds swept;
        float max_speed{};
        bool moving{};

        bool operator==(const InstanceView&) const = default;
    };

    struct InstanceState {
        Transform transform;
        float derivative[12]{};
        float distance_scale{1};
    };

    struct Mesh {
        MeshView data;
        ::cuda::buffer<float3, ::cuda::mr::device_accessible> positions;
        ::cuda::buffer<int3, ::cuda::mr::device_accessible> triangles;
        ::cuda::buffer<int, ::cuda::mr::device_accessible> order;
        ::cuda::buffer<Bounds, ::cuda::mr::device_accessible> nodes;

        Mesh(std::span<const float3> points, std::span<const int3> faces, ::cuda::stream_ref stream, ::cuda::device_memory_pool_ref pool);
        Mesh(const Mesh&)            = delete;
        Mesh& operator=(const Mesh&) = delete;
        Mesh(Mesh&&) noexcept        = default;
        Mesh& operator=(Mesh&&)      = delete;
    };

    struct Scene {
        ::cuda::stream_ref stream;
        ::cuda::device_memory_pool_ref pool;
        std::vector<Mesh> meshes;
        std::vector<Shape> shapes;
        std::vector<Instance> instances;
        std::vector<InstanceView> prepared;

        Scene(::cuda::stream_ref stream, ::cuda::device_memory_pool_ref pool);
        Scene(const Scene&)            = delete;
        Scene& operator=(const Scene&) = delete;
        int add_mesh(std::span<const float3> positions, std::span<const int3> triangles);
        void prepare(float start_time, float end_time);
    };
}
#endif

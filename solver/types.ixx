export module pyro.types;
import std;

export namespace pyro {
    enum struct Shape { sphere, box, mesh };

    struct Triangle {
        std::array<float, 3> a{}, b{}, c{};
    };

    struct Bounds {
        std::array<float, 3> lower{}, upper{};
    };

    struct Motion {
        float matrix[12]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
        float derivative[12]{};
        float time{}, maxSpeed{};
        Bounds swept{};
    };

    struct Source {
        std::array<float, 3> center{};
        std::array<float, 3> halfExtent{0.25f, 0.25f, 0.25f};
        std::array<float, 3> velocity{};
        float smokeRate{1.0f}; // Concentration injected per second.
        float temperatureRate{1.0f};
        float velocityRate{4.0f};
        float startTime{};
        float endTime{1.0e20f};
        Shape shape{Shape::sphere};
        int mesh{-1}; // Index returned by Solver::add_mesh; used only for Shape::mesh.
        Motion motion;
        float inheritVelocity{};
    };

    struct Collider {
        std::array<float, 3> center{};
        std::array<float, 3> halfExtent{0.5f, 0.5f, 0.5f};
        std::array<float, 4> orientation{0.0f, 0.0f, 0.0f, 1.0f}; // Unit quaternion (x, y, z, w) at poseTime.
        std::array<float, 3> linearVelocity{};
        std::array<float, 3> angularVelocity{};
        float poseTime{};
        Shape shape{Shape::sphere};
        int mesh{-1}; // Index returned by Solver::add_mesh; used only for Shape::mesh.
        Motion motion;
        float surfaceOffset{};
    };

} // namespace pyro

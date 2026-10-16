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
        float time{}, max_speed{};
        Bounds swept{};
    };

    struct Source {
        std::array<float, 3> center{};
        std::array<float, 3> half_extent{0.25f, 0.25f, 0.25f};
        std::array<float, 3> velocity{};
        float smoke_rate{1.0f};
        float temperature_rate{1.0f};
        float velocity_rate{4.0f};
        float start_time{};
        float end_time{1.0e20f};
        Shape shape{Shape::sphere};
        int mesh{-1};
        Motion motion;
        float inherit_velocity{};
    };

    struct Collider {
        std::array<float, 3> center{};
        std::array<float, 3> half_extent{0.5f, 0.5f, 0.5f};
        std::array<float, 4> orientation{0.0f, 0.0f, 0.0f, 1.0f};
        std::array<float, 3> linear_velocity{};
        std::array<float, 3> angular_velocity{};
        float pose_time{};
        Shape shape{Shape::sphere};
        int mesh{-1};
        Motion motion;
        float surface_offset{};
    };

} // namespace pyro

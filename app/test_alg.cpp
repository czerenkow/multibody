#include "test_alg.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/vec_swizzle.hpp>

namespace test_alg {
const float GRAVITY = 1.0f;
const float EPS2 = 0.08f;
const float DELTA_T = 0.001f;

// Function calculate Hamiltonian for current state
// This is for testing purposes.
float computeHamiltonian(const Vertex data[], std::size_t N) {
    // Potential Energy
    float U = 0.0f;
    for (std::size_t j = 1; j < N; j++) {
        for (std::size_t i = 0; i < j; i++) {
            auto& vi = data[i];
            auto& vj = data[j];
            const float mass_i = vi.pos_m.w;
            const float mass_j = vj.pos_m.w;
            const glm::vec3 rvi {vi.pos_m.x, vi.pos_m.y, vi.pos_m.z};
            const glm::vec3 rvj {vj.pos_m.x, vj.pos_m.y, vj.pos_m.z};
            auto f = glm::length(rvj - rvi);
            U -= mass_i * mass_j / f;
        }
    }
    U = GRAVITY * U;

    // Kinetic Energy
    float T = 0.0f;
    for (std::size_t j = 0; j < N; j++) {
        auto& vj = data[j];
        glm::vec3 v {vj.velocity.x, vj.velocity.y, vj.velocity.z};
        const float mass = vj.pos_m.w;
        glm::vec3 p = mass * v;
        T += glm::dot(p, p) / (2.0f * mass);
    }

    return T + U;
}



void update(Vertex data[], std::size_t N) {
    auto bodyInteraction_eps2 = [](glm::vec3 bi, glm::vec3 bj, float mj) -> glm::vec3 {
        glm::vec3 r = bj - bi;
        float distSqr = glm::dot(r, r) + EPS2;
        float distSixth = std::sqrt(distSqr * distSqr * distSqr);
        return GRAVITY * r * mj / distSixth;
    };

    auto bodyInteraction_correct = [](glm::vec3 bi, glm::vec3 bj, float mj) -> glm::vec3 {
        glm::vec3 r = bj - bi;
        float distSqr = glm::dot(r, r);// + EPS2;
        float distSixth = std::sqrt(distSqr * distSqr * distSqr);
        return GRAVITY * r * mj / distSixth;
    };

    // calculate acc for each body
    for (size_t i = 0; i < N; i++)
    {
        glm::vec3 a(0.0f);
        glm::vec3 poz_i = glm::xyz(data[i].pos_m);
        for (size_t j = 0; j < N; j++)
        {
            //if (j == i) continue;
            glm::vec3 poz_j = glm::xyz(data[j].pos_m);
            float mass_j = data[j].pos_m.w;
            a += bodyInteraction_eps2(poz_i, poz_j, mass_j);
        }
        data[i].velocity += glm::vec4{DELTA_T * a, 0.0f};
    }

    for (size_t i = 0; i < N; i++) {
        auto& vi = data[i];
        data[i].setPos(vi.pos() + DELTA_T * glm::xyz(vi.velocity));
    }
}

} // namespace: test_alg


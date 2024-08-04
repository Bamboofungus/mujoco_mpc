// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/tasks/gymnasium_swimmer/gymnasium_swimmer.h"

#include <cstdio>
#include <string>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc {
  std::string GymnasiumSwimmer::XmlPath() const {
    return GetModelPath("gymnasium_swimmer/task.xml");
  }
  std::string GymnasiumSwimmer::Name() const { return "GymnasiumSwimmer"; }
  // ----------------- Residuals for gymnasium-swimmer task ----------------
  //   Number of residuals: 3
  //     Residual (0-1): control (Two joints)
  //     Residual (2): velocity (positive-x)
  // -------------------------------------------------------------
  void GymnasiumSwimmer::ResidualFn::Residual(const mjModel* model, const mjData* data,
                         double* residual) const {
    // ---------- Residuals (0-1) ----------
    // controls
    mju_copy(residual, data->ctrl, model->nu);

    // ---------- Residual (2) ----------
    // velocity in positive x direction (negated to reward controller)
    double* torso_linvel = SensorByName(model, data, "torso_linvel");
    double torso_linvel_x = torso_linvel[1];
    residual[2] = -torso_linvel_x - 100;
    // printf("Residual: %f\n", residual[2]);
  }
}  // namespace mjpc

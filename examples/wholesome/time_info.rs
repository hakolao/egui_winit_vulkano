// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::time::Instant;

const NANOS_PER_MILLI: f32 = 1_000_000f32;

pub struct TimeInfo {
    dt: f32,
    fps: f32,
    frame_sum: f32,
    dt_sum: f32,
    prev_time: Instant,
}

impl TimeInfo {
    pub fn new() -> TimeInfo {
        TimeInfo { dt: 0.0, fps: 0.0, frame_sum: 0.0, dt_sum: 0.0, prev_time: Instant::now() }
    }

    #[allow(dead_code)]
    pub fn dt(&self) -> f32 {
        self.dt
    }

    pub fn fps(&self) -> f32 {
        self.fps
    }

    pub fn update(&mut self) {
        let now = Instant::now();
        self.frame_sum += 1.0;
        // Assume duration is never over full second, so ignore whole seconds in Duration
        self.dt = now.duration_since(self.prev_time).subsec_nanos() as f32 / NANOS_PER_MILLI;
        self.dt_sum += self.dt;
        if self.dt_sum >= 1000.0 {
            self.fps = 1000.0 / (self.dt_sum / self.frame_sum);
            self.dt_sum = 0.0;
            self.frame_sum = 0.0;
        }
        self.prev_time = now;
    }
}

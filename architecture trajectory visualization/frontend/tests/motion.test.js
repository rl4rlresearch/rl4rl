import test from "node:test";
import assert from "node:assert/strict";
import { MorphSpring, nodeScale } from "../src/motion.js";

test("morphing eases into motion, settles exactly, and is frame-rate independent", () => {
  const slow = new MorphSpring([0, 1], 0),
    fast = new MorphSpring([0, 1], 0);
  for (const spring of [slow, fast]) spring.retarget([12, 0], 0);
  slow.step(1);
  assert.ok(slow.values[0] > 0 && slow.values[0] < 0.01);
  for (let t = 50; t <= 500; t += 50) slow.step(t);
  for (let t = 10; t <= 500; t += 10) fast.step(t);
  assert.ok(Math.abs(slow.values[0] - fast.values[0]) < 1e-9);
  assert.ok(slow.values[0] < 12 && slow.values[1] > 0);
  slow.step(3000);
  assert.deepEqual(slow.values, [12, 0]);
  assert.equal(slow.active, false);
});

test("a new scrub target retains the in-flight pose and velocity without jumping", () => {
  const spring = new MorphSpring([0, 0.08], 0);
  spring.retarget([20, 1], 0);
  spring.step(180);
  const pose = [...spring.values],
    velocity = [...spring.velocity];
  spring.retarget([-5, 0.3], 180);
  assert.deepEqual(spring.values, pose);
  assert.deepEqual(spring.velocity, velocity);
  spring.step(181);
  assert.ok(Math.abs(spring.values[0] - pose[0]) < 0.2);
  spring.step(4000);
  assert.deepEqual(spring.values, [-5, 0.3]);
});

test("reduced motion and long background pauses settle with no leftover transitions", () => {
  const spring = new MorphSpring([1, 0, 3], 0);
  spring.retarget([2, 1, 8], 0);
  spring.step(10, true);
  assert.deepEqual(spring.values, [2, 1, 8]);
  assert.deepEqual(spring.velocity, [0, 0, 0]);
  assert.equal(spring.active, false);
  spring.retarget([4, 0, 2], 20);
  spring.step(1e6);
  assert.deepEqual(spring.values, [4, 0, 2]);
});

test("schematic size changes use explicit numeric dimensions without inferring expressions", () => {
  const small = nodeScale({ config: { out_channels: 32 } });
  const large = nodeScale({ config: { out_channels: 128 } });
  assert.ok(large[0] > small[0]);
  assert.deepEqual(
    nodeScale({ config: { out_channels: { expression: "width * 2" } } }),
    [1, 1, 1],
  );
  assert.deepEqual(nodeScale({ config: { out_channels: "128" } }), [1, 1, 1]);
  assert.ok(nodeScale({ config: { out_features: 1e20 } })[0] <= 1.55);
});

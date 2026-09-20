/** Analytic critically damped motion: continuous position/velocity on retarget. */
export class MorphSpring {
  constructor(values, now = 0, durationMs = 850) {
    this.values = [...values];
    this.target = [...values];
    this.velocity = values.map(() => 0);
    this.time = now;
    this.durationMs = durationMs;
    this.active = false;
  }
  retarget(target, now, durationMs = this.durationMs) {
    this.step(now);
    this.target = [...target];
    this.durationMs = Math.max(80, durationMs);
    this.active = this.target.some(
      (value, i) =>
        Math.abs(value - this.values[i]) > 0.0001 ||
        Math.abs(this.velocity[i]) > 0.0001,
    );
  }
  step(now, immediate = false) {
    const elapsed = Math.max(0, (now - this.time) / 1000);
    this.time = now;
    if (immediate) {
      this.values = [...this.target];
      this.velocity.fill(0);
      this.active = false;
      return false;
    }
    if (!this.active) return false;
    const omega = 9000 / this.durationMs;
    const decay = Math.exp(-omega * elapsed);
    let active = false;
    for (let i = 0; i < this.values.length; i++) {
      const displacement = this.values[i] - this.target[i];
      const coefficient = this.velocity[i] + omega * displacement;
      const offset = (displacement + coefficient * elapsed) * decay;
      const velocity =
        (this.velocity[i] - omega * coefficient * elapsed) * decay;
      if (Math.abs(offset) < 0.0005 && Math.abs(velocity) < 0.003) {
        this.values[i] = this.target[i];
        this.velocity[i] = 0;
      } else {
        this.values[i] = this.target[i] + offset;
        this.velocity[i] = velocity;
        active = true;
      }
    }
    this.active = active;
    return active;
  }
}

/** Bounded, schematic dimensions; only numeric source values affect geometry. */
export function nodeScale(node) {
  const config = node.config || {};
  const dimension = [
    config.out_features,
    config.out_channels,
    config.hidden_size,
    config.embed_dim,
    config.d_model,
  ].find((n) => typeof n === "number" && Number.isFinite(n) && n > 0);
  const width =
    dimension === undefined
      ? 1
      : Math.max(0.85, Math.min(1.55, 0.7 + Math.log2(dimension + 1) * 0.085));
  const components = config.components;
  const height =
    typeof components === "number" && components > 1
      ? 1 + Math.min(0.6, Math.log2(components) * 0.12)
      : 1;
  return [width, height, 1];
}

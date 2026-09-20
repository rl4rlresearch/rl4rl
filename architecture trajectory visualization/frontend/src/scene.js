import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  CSS2DRenderer,
  CSS2DObject,
} from "three/addons/renderers/CSS2DRenderer.js";
import { MorphSpring, nodeScale } from "./motion.js";

export const PALETTE = {
  embedding: "#95bea4",
  token_embedding: "#95bea4",
  attention: "#aaa2ca",
  feed_forward: "#d4b87b",
  linear: "#d4b87b",
  normalization: "#99b9c4",
  convolution: "#d39588",
  pooling: "#c4ac8d",
  recurrent: "#bd9bbb",
  routing: "#bd9bbb",
  readout: "#b3c5a7",
  input: "#a8b4aa",
  custom: "#9baba1",
  module: "#9baba1",
  algebraic: "#b8bf99",
  positional: "#c2bcb0",
  composition: "#aabda6",
  state: "#ba99b2",
};
export const colorFor = (kind) => PALETTE[kind] || "#9baba1";
const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const title = (x) => String(x || "component").replaceAll("_", " ");
const dimension = (value) =>
  typeof value === "object" && value !== null
    ? String(value.expression ?? value.value ?? "symbolic")
    : String(value);
let nextVisualId = 0;

export class ArchitectureScene {
  constructor(mount, onSelect) {
    this.mount = mount;
    this.onSelect = onSelect;
    this.graph = null;
    this.objects = new Map();
    this.retiring = new Set();
    this.connections = new Map();
    this.footprints = new Map();
    this.durationMs = 850;
    this.frame = 0;
    this.fitted = false;
    this.top = false;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#161a19");
    this.camera = new THREE.PerspectiveCamera(37, 1, 0.05, 5000);
    this.camera.position.set(18, 19, 26);
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      preserveDrawingBuffer: true,
    });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.setClearColor("#161a19");
    mount.append(this.renderer.domElement);
    this.labels = new CSS2DRenderer();
    Object.assign(this.labels.domElement.style, {
      position: "absolute",
      inset: "0",
      pointerEvents: "none",
    });
    mount.append(this.labels.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.16;
    this.controls.minDistance = 3;
    this.controls.maxDistance = 900;
    this.controls.addEventListener("change", () => this.requestRender());
    this.scene.add(new THREE.AmbientLight("#e5f3e8", 2));
    const light = new THREE.DirectionalLight("#ffffff", 2.7);
    light.position.set(4, 18, 12);
    this.scene.add(light);
    const fill = new THREE.DirectionalLight("#b4c8bc", 0.9);
    fill.position.set(-8, 2, -5);
    this.scene.add(fill);
    this.model = new THREE.Group();
    this.scene.add(this.model);
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.renderer.domElement.addEventListener("pointerdown", (e) => {
      this.down = { x: e.clientX, y: e.clientY };
    });
    this.renderer.domElement.addEventListener("pointerup", (e) => {
      if (
        !this.down ||
        Math.hypot(e.clientX - this.down.x, e.clientY - this.down.y) > 5
      )
        return;
      const r = this.renderer.domElement.getBoundingClientRect();
      this.pointer.set(
        ((e.clientX - r.left) / r.width) * 2 - 1,
        (-(e.clientY - r.top) / r.height) * 2 + 1,
      );
      this.raycaster.setFromCamera(this.pointer, this.camera);
      const hit = this.raycaster.intersectObjects(
        [...this.objects.values()].map((x) => x.mesh),
        false,
      )[0];
      if (hit) this.select(hit.object.userData.node.id);
    });
    this.renderer.domElement.addEventListener("pointermove", (e) => {
      if (e.buttons) return;
      const r = this.renderer.domElement.getBoundingClientRect();
      this.pointer.set(
        ((e.clientX - r.left) / r.width) * 2 - 1,
        (-(e.clientY - r.top) / r.height) * 2 + 1,
      );
      this.raycaster.setFromCamera(this.pointer, this.camera);
      const hit = this.raycaster.intersectObjects(
        [...this.objects.values()].map((o) => o.mesh),
        false,
      )[0];
      this.renderer.domElement.style.cursor = hit ? "pointer" : "grab";
      this.renderer.domElement.title = hit
        ? `${hit.object.userData.node.label} · ${title(hit.object.userData.node.kind)} · click for evidence`
        : "";
      const hovered = hit?.object.userData.node.id;
      if (hovered !== this.hovered) {
        this.hovered = hovered;
        this.requestRender();
      }
    });
    this.resize = new ResizeObserver(() => this.resizeScene());
    this.resize.observe(mount);
    this.resizeScene();
  }
  resizeScene() {
    const { width, height } = this.mount.getBoundingClientRect();
    if (!width || !height) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
    this.labels.setSize(width, height);
    this.requestRender();
  }
  requestRender() {
    if (!this.disposed && !this.frame)
      this.frame = requestAnimationFrame((t) => this.render(t));
  }
  render(t) {
    this.frame = 0;
    const active = this.advance(t, reduced());
    this.mount.dataset.transition = active ? "active" : "";
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.labels.render(this.scene, this.camera);
    this.placeLabels();
    if (active) this.requestRender();
  }
  advance(now, immediate = false) {
    let active = false;
    for (const o of [...this.objects.values(), ...this.retiring]) {
      active = o.motion.step(now, immediate) || active;
      const v = o.motion.values;
      o.container.position.set(v[0], v[1], v[2]);
      o.mesh.scale.set(
        Math.max(0.01, v[3]),
        Math.max(0.01, v[4]),
        Math.max(0.01, v[5]),
      );
      o.mesh.material.color.setRGB(v[6], v[7], v[8]);
      const alpha = Math.max(0, Math.min(1, v[9]));
      o.mesh.material.opacity = alpha;
      o.mesh.material.depthWrite = alpha > 0.95;
      o.outline.material.color.setRGB(v[10], v[11], v[12]);
      o.outline.material.opacity = alpha * 0.65;
      o.label.element.style.opacity = String(alpha);
      o.label.element.style.pointerEvents = this.retiring.has(o) ? "none" : "";
      for (const line of o.dividers) line.material.opacity = alpha;
      if (this.retiring.has(o) && !o.motion.active) {
        this.retiring.delete(o);
        this.disposeNode(o);
      }
    }
    for (const [key, edge] of this.connections) {
      active = edge.fade.step(now, immediate) || active;
      if (!edge.wanted && !edge.fade.active) {
        this.disposeObject(edge.line);
        if (edge.arrow) this.disposeObject(edge.arrow);
        this.connections.delete(key);
        continue;
      }
      this.updateConnection(edge);
    }
    for (const [key, group] of this.footprints) {
      active = group.motion.step(now, immediate) || active;
      if (!group.wanted && !group.motion.active) {
        this.disposeObject(group.object);
        this.footprints.delete(key);
        continue;
      }
      const v = group.motion.values;
      group.object.position.set(v[0], v[1], v[2]);
      group.object.scale.set(Math.max(0.1, v[3]), 0.04, Math.max(0.1, v[4]));
      group.object.material.opacity = Math.max(0, v[5]) * 0.28;
    }
    return active;
  }
  placeLabels() {
    const occupied = [];
    const priority = (o) =>
      (o.node.id === this.hovered || o.node.id === this.selected ? 10 : 0) +
      (o.label.element.classList.contains("changed") ||
      o.label.element.classList.contains("added")
        ? 3
        : 0) +
      (o.node.is_output || o.node.kind === "input" ? 1 : 0) +
      (o.labelVisible ? 0.2 : 0);
    const overlaps = (rect) =>
      occupied.some(
        (r) =>
          rect.left < r.right + 4 &&
          rect.right + 4 > r.left &&
          rect.top < r.bottom + 4 &&
          rect.bottom + 4 > r.top,
      );
    for (const o of [...this.objects.values()].sort(
      (a, b) => priority(b) - priority(a),
    )) {
      const rect = o.label.element.getBoundingClientRect();
      o.labelVisible = !overlaps(rect);
      o.label.element.style.visibility = o.labelVisible ? "visible" : "hidden";
      if (o.labelVisible) occupied.push(rect);
    }
    for (const o of this.retiring)
      o.label.element.style.visibility = overlaps(
        o.label.element.getBoundingClientRect(),
      )
        ? "hidden"
        : "visible";
  }
  disposeObject(object) {
    object.removeFromParent();
    object.traverse((child) => {
      child.geometry?.dispose();
      if (Array.isArray(child.material))
        child.material.forEach((m) => m.dispose());
      else child.material?.dispose();
    });
  }
  disposeNode(o) {
    o.label.element.remove();
    this.disposeObject(o.container);
  }
  clear() {
    for (const o of new Set([...this.objects.values(), ...this.retiring]))
      this.disposeNode(o);
    this.objects.clear();
    this.retiring.clear();
    for (const edge of this.connections.values()) {
      this.disposeObject(edge.line);
      if (edge.arrow) this.disposeObject(edge.arrow);
    }
    this.connections.clear();
    for (const group of this.footprints.values())
      this.disposeObject(group.object);
    this.footprints.clear();
    this.graph = null;
    this.mount.dataset.transition = "";
    this.requestRender();
  }
  createNode(n, values, now) {
    const container = new THREE.Group();
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(4, 0.7, 1.55),
      new THREE.MeshStandardMaterial({
        color: colorFor(n.kind),
        roughness: 0.72,
        metalness: 0.02,
        transparent: true,
      }),
    );
    const outline = new THREE.LineSegments(
      new THREE.EdgesGeometry(mesh.geometry),
      new THREE.LineBasicMaterial({
        color: "#d4dfd3",
        transparent: true,
        opacity: 0.65,
      }),
    );
    mesh.add(outline);
    container.add(mesh);
    const e = document.createElement("button");
    e.type = "button";
    const name = document.createTextNode(""),
      sub = document.createElement("small");
    e.append(name, sub);
    const label = new CSS2DObject(e);
    label.position.set(0, -0.9, 0);
    label.center.set(0.5, 0);
    container.add(label);
    const o = {
      uid: ++nextVisualId,
      container,
      mesh,
      outline,
      label,
      name,
      sub,
      node: n,
      dividers: [],
      heads: 0,
      motion: new MorphSpring(values, now, this.durationMs),
    };
    e.addEventListener("click", () => this.select(o.node.id));
    this.model.add(container);
    return o;
  }
  updateNode(o, n, status) {
    o.node = n;
    o.mesh.userData.node = n;
    o.name.nodeValue = n.label || title(n.kind);
    const e = o.label.element;
    e.className = "node-label " + (status || "");
    e.dataset.nodeId = n.id;
    e.dataset.visualId = String(o.uid);
    e.title = `${n.label || n.id} · ${title(n.kind)} — inspect source evidence`;
    e.setAttribute("aria-label", e.title);
    const c = n.config || {};
    let dim = title(n.kind);
    if (c.in_features != null && c.out_features != null)
      dim = `${dimension(c.in_features)} → ${dimension(c.out_features)}`;
    else if (c.in_channels != null && c.out_channels != null)
      dim = `${dimension(c.in_channels)} → ${dimension(c.out_channels)} channels`;
    else if (c.input_size != null && c.hidden_size != null)
      dim = `${dimension(c.input_size)} → ${dimension(c.hidden_size)} state`;
    else if (c.num_heads != null) dim = `${dimension(c.num_heads)} heads`;
    else if (n.repeat) dim = `× ${dimension(n.repeat)}`;
    o.sub.textContent = dim;
    const heads =
      n.kind === "attention" &&
      Number.isInteger(c.num_heads) &&
      c.num_heads > 1 &&
      c.num_heads <= 16
        ? c.num_heads
        : 0;
    if (o.heads !== heads) {
      for (const line of o.dividers) this.disposeObject(line);
      o.dividers = [];
      for (let head = 1; head < heads; head++) {
        const x = -2 + (4 * head) / heads;
        const line = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(x, 0.36, -0.775),
            new THREE.Vector3(x, 0.36, 0.775),
          ]),
          new THREE.LineBasicMaterial({ color: "#514b64", transparent: true }),
        );
        o.mesh.add(line);
        o.dividers.push(line);
      }
      o.heads = heads;
    }
  }
  nodeTarget(n, status) {
    const color = new THREE.Color(colorFor(n.kind));
    const outline = new THREE.Color(
      status === "added"
        ? "#d2efdc"
        : status === "changed"
          ? "#ffdd8f"
          : "#d4dfd3",
    );
    return [
      n.x,
      n.y,
      n.z,
      ...nodeScale(n),
      color.r,
      color.g,
      color.b,
      1,
      outline.r,
      outline.g,
      outline.b,
    ];
  }
  setGraph(
    graph,
    layout,
    diff,
    transitionDiff = diff,
    { durationMs = 850 } = {},
  ) {
    const now = performance.now();
    this.advance(now, reduced());
    this.durationMs = Math.max(120, Math.min(1800, durationMs));
    const previous = this.objects,
      wasEmpty = previous.size === 0;
    const matched = new Map(
      (transitionDiff?.matches || []).map((m) => [
        m.after ?? m.after_id,
        m.before ?? m.before_id,
      ]),
    );
    this.graph = graph;
    this.layout = layout;
    this.objects = new Map();
    const reused = new Set();
    for (const n of layout.nodes || []) {
      const oldId = transitionDiff ? matched.get(n.id) : n.id;
      let o = oldId == null ? null : previous.get(oldId);
      if (o && reused.has(o)) o = null;
      const status = diff?.nodeStatus?.[n.id],
        target = this.nodeTarget(n, status);
      if (!o) {
        const initial = [...target];
        if (!wasEmpty && !reduced()) {
          // This origin is only a visual transition; it adds no graph relationship.
          const neighbors = new Set(
            (graph.edges || []).flatMap((e) =>
              e.source === n.id
                ? [e.target]
                : e.target === n.id
                  ? [e.source]
                  : [],
            ),
          );
          const anchors = (layout.nodes || []).filter(
            (c) =>
              matched.has(c.id) &&
              (neighbors.has(c.id) || c.group_id === n.group_id),
          );
          let closest = null,
            distance = Infinity;
          for (const candidate of anchors) {
            const object = previous.get(matched.get(candidate.id));
            if (!object) continue;
            const d = object.container.position.distanceToSquared(
              new THREE.Vector3(n.x, n.y, n.z),
            );
            if (d < distance) {
              closest = object;
              distance = d;
            }
          }
          if (closest)
            for (let axis = 0; axis < 3; axis++)
              initial[axis] =
                target[axis] * 0.4 + closest.motion.values[axis] * 0.6;
          initial[3] *= 0.08;
          initial[4] *= 0.08;
          initial[5] *= 0.08;
          initial[9] = 0;
        }
        o = this.createNode(n, initial, now);
      }
      reused.add(o);
      this.updateNode(o, n, status);
      o.motion.retarget(target, now, this.durationMs);
      this.objects.set(n.id, o);
    }
    for (const o of previous.values()) {
      if (reused.has(o)) continue;
      this.retiring.add(o);
      o.label.element.className = "node-label departing";
      o.label.element.inert = true;
      const target = [...o.motion.values];
      target[3] = target[4] = target[5] = 0.08;
      target[9] = 0;
      o.motion.retarget(target, now, this.durationMs * 0.85);
    }
    this.syncConnections(graph, now, wasEmpty);
    this.syncGroups(layout.nodes || [], now, wasEmpty);
    this.advance(now, reduced());
    if (!this.fitted && this.objects.size) {
      this.fit();
      this.fitted = true;
    }
    this.mount.dataset.transition = this.isMoving() ? "active" : "";
    this.requestRender();
  }
  syncConnections(graph, now, wasEmpty) {
    for (const edge of this.connections.values()) edge.wanted = false;
    for (const edge of graph.edges || []) {
      if (edge.type === "containment") continue;
      const source = this.objects.get(edge.source),
        target = this.objects.get(edge.target);
      if (!source || !target) continue;
      const key = JSON.stringify([
        source.uid,
        target.uid,
        edge.type,
        edge.source_port,
        edge.target_port,
      ]);
      let item = this.connections.get(key);
      if (!item) {
        const shared = /shar|tie/.test(edge.type),
          loop = /recur|state/.test(edge.type) || source === target;
        const material = shared
          ? new THREE.LineDashedMaterial({
              color: "#d6bf7f",
              dashSize: 0.2,
              gapSize: 0.15,
              transparent: true,
            })
          : new THREE.LineBasicMaterial({
              color: loop ? "#c1a4c2" : "#729883",
              transparent: true,
            });
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute(
          "position",
          new THREE.BufferAttribute(new Float32Array(31 * 3), 3),
        );
        const line = new THREE.Line(geometry, material);
        this.model.add(line);
        let arrow = null;
        if (!shared && source !== target) {
          arrow = new THREE.ArrowHelper(
            new THREE.Vector3(1, 0, 0),
            new THREE.Vector3(),
            0.4,
            "#8aaa97",
            0.23,
            0.13,
          );
          arrow.line.material.transparent =
            arrow.cone.material.transparent = true;
          // ArrowHelper shares default geometries; own them for safe disposal.
          arrow.line.geometry = arrow.line.geometry.clone();
          arrow.cone.geometry = arrow.cone.geometry.clone();
          this.model.add(arrow);
        }
        const points = Array.from(
          { length: source === target ? 4 : 3 },
          () => new THREE.Vector3(),
        );
        item = {
          source,
          target,
          shared,
          loop,
          line,
          arrow,
          points,
          curve: new THREE.CatmullRomCurve3(points),
          sample: new THREE.Vector3(),
          fade: new MorphSpring([wasEmpty ? 1 : 0], now, this.durationMs),
        };
        this.connections.set(key, item);
      }
      item.wanted = true;
      item.fade.retarget([1], now, this.durationMs);
    }
    for (const edge of this.connections.values())
      if (!edge.wanted) edge.fade.retarget([0], now, this.durationMs * 0.7);
  }
  updateConnection(edge) {
    const a = edge.source.container.position,
      b = edge.target.container.position,
      p = edge.points;
    p[0].copy(a);
    p[p.length - 1].copy(b);
    if (edge.source === edge.target) {
      p[1].copy(a).add(new THREE.Vector3(1.6, 2, 0));
      p[2].copy(a).add(new THREE.Vector3(-1.6, 2, 0));
    } else {
      p[1].copy(a).lerp(b, 0.5);
      p[1].y += edge.shared
        ? 1.7
        : edge.loop
          ? 3
          : Math.max(0.3, Math.abs(a.z - b.z) * 0.12);
      if (edge.loop) p[1].z += 3;
    }
    const attr = edge.line.geometry.attributes.position;
    for (let i = 0; i <= 30; i++) {
      edge.curve.getPoint(i / 30, edge.sample);
      attr.setXYZ(i, edge.sample.x, edge.sample.y, edge.sample.z);
    }
    attr.needsUpdate = true;
    edge.line.geometry.computeBoundingSphere();
    if (edge.shared) edge.line.computeLineDistances();
    const alpha = Math.max(
      0,
      Math.min(
        edge.fade.values[0],
        edge.source.motion.values[9],
        edge.target.motion.values[9],
      ),
    );
    edge.line.material.opacity = alpha * (edge.shared ? 0.8 : 0.7);
    if (edge.arrow) {
      edge.arrow.position.copy(edge.curve.getPoint(0.9, edge.sample));
      edge.arrow.setDirection(edge.curve.getTangent(0.9, edge.sample));
      edge.arrow.line.material.opacity = edge.arrow.cone.material.opacity =
        alpha * 0.8;
    }
  }
  syncGroups(nodes, now, wasEmpty) {
    for (const group of this.footprints.values()) group.wanted = false;
    const groups = new Map();
    for (const n of nodes)
      if (n.group_id) {
        if (!groups.has(n.group_id)) groups.set(n.group_id, []);
        groups.get(n.group_id).push(new THREE.Vector3(n.x, n.y, n.z));
      }
    for (const [id, points] of groups) {
      if (points.length < 2) continue;
      const box = new THREE.Box3()
        .setFromPoints(points)
        .expandByVector(new THREE.Vector3(2.3, 0.3, 1));
      const center = box.getCenter(new THREE.Vector3()),
        size = box.getSize(new THREE.Vector3()),
        target = [center.x, box.min.y - 0.4, center.z, size.x, size.z, 1];
      let group = this.footprints.get(id);
      if (!group) {
        const object = new THREE.LineSegments(
          new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
          new THREE.LineBasicMaterial({
            color: "#466052",
            transparent: true,
            opacity: 0,
          }),
        );
        this.model.add(object);
        const initial = [...target];
        initial[5] = wasEmpty ? 1 : 0;
        group = {
          object,
          motion: new MorphSpring(initial, now, this.durationMs),
        };
        this.footprints.set(id, group);
      }
      group.wanted = true;
      group.motion.retarget(target, now, this.durationMs);
    }
    for (const group of this.footprints.values())
      if (!group.wanted) {
        const target = [...group.motion.target];
        target[5] = 0;
        group.motion.retarget(target, now, this.durationMs * 0.7);
      }
  }
  isMoving() {
    return (
      [...this.objects.values(), ...this.retiring].some(
        (o) => o.motion.active,
      ) ||
      [...this.connections.values()].some((e) => e.fade.active) ||
      [...this.footprints.values()].some((g) => g.motion.active)
    );
  }
  inspectMotion() {
    let attachmentError = 0;
    for (const edge of this.connections.values()) {
      const p = edge.line.geometry.attributes.position;
      attachmentError = Math.max(
        attachmentError,
        new THREE.Vector3()
          .fromBufferAttribute(p, 0)
          .distanceTo(edge.source.container.position),
        new THREE.Vector3()
          .fromBufferAttribute(p, 30)
          .distanceTo(edge.target.container.position),
      );
    }
    return {
      active: this.isMoving(),
      retiring: this.retiring.size,
      connections: this.connections.size,
      attachmentError,
      camera: this.camera.position.toArray(),
      nodes: [...this.objects.values()].map((o) => ({
        id: o.node.id,
        uid: o.uid,
        position: o.container.position.toArray(),
        scale: o.mesh.scale.toArray(),
        opacity: o.mesh.material.opacity,
        target: o.motion.target.slice(0, 6),
      })),
    };
  }
  select(id) {
    this.selected = id;
    for (const [key, o] of this.objects) {
      o.mesh.material.emissive.set(key === id ? "#233b2b" : "#000000");
      o.label.element.style.outline = key === id ? "1px solid #e2efdf" : "";
    }
    const n = this.objects.get(id)?.node;
    if (n) this.onSelect(n);
    this.requestRender();
  }
  fit(top = this.top) {
    if (!this.objects.size) return;
    // A breakpoint can request a new graph before ResizeObserver has fired.
    this.resizeScene();
    this.top = top;
    const box = new THREE.Box3();
    for (const o of this.objects.values())
      box.expandByPoint(new THREE.Vector3(o.node.x, o.node.y, o.node.z));
    box.expandByVector(new THREE.Vector3(2, 2, 2));
    const center = box.getCenter(new THREE.Vector3());
    const direction = top
      ? new THREE.Vector3(0.02, 1, 0.01).normalize()
      : new THREE.Vector3(0.18, 2.4, 1).normalize();
    const right = new THREE.Vector3()
      .crossVectors(this.camera.up, direction)
      .normalize();
    const up = new THREE.Vector3().crossVectors(direction, right).normalize();
    const tangent = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
    let distance = 7;
    // Fit every corner in camera space, accounting for perspective depth.
    for (const x of [box.min.x, box.max.x])
      for (const y of [box.min.y, box.max.y])
        for (const z of [box.min.z, box.max.z]) {
          const corner = new THREE.Vector3(x, y, z).sub(center);
          const depth = corner.dot(direction);
          distance = Math.max(
            distance,
            depth + Math.abs(corner.dot(up)) / tangent,
            depth +
              Math.abs(corner.dot(right)) / (tangent * this.camera.aspect),
          );
        }
    distance *= 1.14;
    this.camera.position.copy(center).addScaledVector(direction, distance);
    this.controls.target.copy(center);
    this.controls.update();
    this.requestRender();
  }
  capture({ title: heading, subtitle, metric, provenance }) {
    // A still represents the selected recorded snapshot, not a morph frame.
    this.advance(performance.now(), true);
    this.mount.dataset.transition = "";
    this.renderer.render(this.scene, this.camera);
    this.labels.render(this.scene, this.camera);
    this.placeLabels();
    const source = this.renderer.domElement;
    const scale = 2,
      w = 1200,
      h =
        Math.max(
          650,
          Math.round((w * source.clientHeight) / source.clientWidth),
        ) + 140;
    const c = document.createElement("canvas");
    c.width = w * scale;
    c.height = h * scale;
    const ctx = c.getContext("2d");
    ctx.scale(scale, scale);
    ctx.fillStyle = "#161a19";
    ctx.fillRect(0, 0, w, h);
    ctx.drawImage(source, 0, 85, w, h - 140);
    ctx.fillStyle = "#e4e9e6";
    ctx.font = "24px system-ui";
    ctx.fillText(heading, 30, 36);
    ctx.font = "13px system-ui";
    ctx.fillStyle = "#a6b8ad";
    ctx.fillText(subtitle.slice(0, 130), 30, 62);
    for (const o of this.objects.values()) {
      if (o.label.element.style.visibility === "hidden") continue;
      const pos = o.container.position
        .clone()
        .add(new THREE.Vector3(0, -0.9, 0))
        .project(this.camera);
      if (pos.z > 1 || Math.abs(pos.x) > 1 || Math.abs(pos.y) > 1) continue;
      const x = ((pos.x + 1) / 2) * w,
        y = 85 + ((1 - pos.y) / 2) * (h - 140);
      ctx.font = "12px system-ui";
      ctx.textAlign = "center";
      ctx.fillStyle = "#e1e9e2";
      ctx.fillText((o.node.label || o.node.kind).slice(0, 35), x, y + 12);
    }
    ctx.textAlign = "left";
    ctx.fillStyle = "#d2dfd4";
    ctx.font = "14px system-ui";
    ctx.fillText(metric, 30, h - 30);
    ctx.textAlign = "right";
    ctx.fillStyle = "#98aa9f";
    ctx.font = "11px system-ui";
    ctx.fillText(provenance.slice(0, 125), w - 30, h - 30);
    return c.toDataURL("image/png");
  }
  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.frame);
    this.resize.disconnect();
    this.controls.dispose();
    this.clear();
    this.renderer.dispose();
    this.renderer.domElement.remove();
    this.labels.domElement.remove();
  }
}

"""
Model the FlyRunner fly in Blender and export it as public/models/fly.glb.

    blender -b --python game/scripts/blender_fly.py -- [--preview out.png]

Conventions: Blender Z-up, the fly faces -Y (which the glTF exporter turns into +Z, the
direction of travel in three.js). Node names are the contract with src/render/fly-model.ts:
body parts are children of "root"; legs are "leg{L|R}{0..2}" (origin at the hip) with a child
"…_tibia" (origin at the knee); wings are "wingL"/"wingR" with the origin at the wing root.
"""
import bpy, bmesh, math, sys, random
from mathutils import Vector, Matrix

random.seed(7)
OUT = "game/public/models/fly.glb"
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
PREVIEW = argv[argv.index("--preview") + 1] if "--preview" in argv else None

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


# ----------------------------------------------------------------------------- helpers
def material(name, color, rough=0.5, metal=0.0, alpha=1.0, emission=None, strength=0.6):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if alpha < 1.0:
        b.inputs["Alpha"].default_value = alpha
        if hasattr(m, "surface_render_method"): m.surface_render_method = "BLENDED"
        if hasattr(m, "blend_method"): m.blend_method = "BLEND"
    if emission:
        b.inputs["Emission Color"].default_value = (*emission, 1.0)
        b.inputs["Emission Strength"].default_value = strength
    return m


CHITIN = material("chitin", (0.16, 0.10, 0.05), 0.45, 0.3)
THORAX = material("thorax", (0.24, 0.16, 0.08), 0.5, 0.25)
STRIPE = material("stripe", (0.46, 0.32, 0.15), 0.45, 0.25)
LEG = material("leg", (0.10, 0.06, 0.03), 0.6, 0.1)
EYE = material("eye", (0.70, 0.10, 0.12), 0.3, 0.1, emission=(0.25, 0.02, 0.03))
HAIR = material("hair", (0.08, 0.05, 0.03), 0.7)
MEMBRANE = material("membrane", (0.85, 0.90, 1.0), 0.1, 0.0, alpha=0.35)
VEIN = material("vein", (0.20, 0.14, 0.08), 0.6)


def finish(obj, name, mats, smooth=True, subdiv=1, parent=None):
    obj.name = name; obj.data.name = name
    obj.data.materials.clear()
    for m in mats: obj.data.materials.append(m)
    if smooth:
        for p in obj.data.polygons: p.use_smooth = True
    if subdiv:
        md = obj.modifiers.new("subd", "SUBSURF"); md.levels = subdiv; md.render_levels = subdiv
    if parent is not None:
        obj.parent = parent; obj.matrix_parent_inverse = parent.matrix_world.inverted()
    return obj


def sphere(name, r, loc, scale=(1, 1, 1), mats=(CHITIN,), seg=24, rings=16, subdiv=1, parent=None, smooth=True):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, segments=seg, ring_count=rings, location=loc)
    o = bpy.context.active_object; o.scale = scale
    bpy.ops.object.transform_apply(scale=True)
    return finish(o, name, mats, smooth, subdiv, parent)


def icosphere(name, r, loc, scale=(1, 1, 1), mats=(CHITIN,), sub=3, parent=None):
    bpy.ops.mesh.primitive_ico_sphere_add(radius=r, subdivisions=sub, location=loc)
    o = bpy.context.active_object; o.scale = scale
    bpy.ops.object.transform_apply(scale=True)
    return finish(o, name, mats, smooth=False, subdiv=0, parent=parent)


def tube(name, a, b, r0, r1, mats=(LEG,), verts=8, parent=None, origin=None):
    """Tapered cylinder from a to b, origin at `origin` (default a)."""
    a, b = Vector(a), Vector(b); d = b - a; L = d.length
    bpy.ops.mesh.primitive_cone_add(vertices=verts, radius1=r0, radius2=r1, depth=L, location=(a + b) / 2)
    o = bpy.context.active_object
    o.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    bpy.ops.object.transform_apply(rotation=True)
    set_origin(o, origin if origin is not None else a)
    return finish(o, name, mats, smooth=True, subdiv=0, parent=parent)


def set_origin(o, p):
    scene.cursor.location = Vector(p)
    bpy.ops.object.select_all(action="DESELECT"); o.select_set(True); bpy.context.view_layer.objects.active = o
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")


def join(objs, name):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join(); o = bpy.context.active_object; o.name = name; o.data.name = name
    return o


# ----------------------------------------------------------------------------- body
bpy.ops.object.empty_add(location=(0, 0, 0)); root = bpy.context.active_object; root.name = "root"

# abdomen: elongated sphere tapered toward the tail, striped by material bands
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.23, segments=40, ring_count=32, location=(0, 0.34, 0.45))
abd = bpy.context.active_object; abd.rotation_euler = (math.pi / 2, 0, 0); bpy.ops.object.transform_apply(rotation=True)
abd.scale = (1.0, 1.8, 0.82); bpy.ops.object.transform_apply(scale=True); finish(abd, "abdomen", (CHITIN, STRIPE), True, 1, root)
bm = bmesh.new(); bm.from_mesh(abd.data)
for v in bm.verts:
    t = max(0.0, (v.co.y - 0.34) / 0.41)           # 0 at the middle, 1 at the tail
    k = 1.0 - 0.45 * t * t
    v.co.x *= k; v.co.z = 0.45 + (v.co.z - 0.45) * k
    t2 = max(0.0, (0.34 - v.co.y) / 0.41)          # front squashes into the thorax
    v.co.x *= 1.0 - 0.15 * t2 * t2
for f in bm.faces:
    y = sum(v.co.y for v in f.verts) / len(f.verts)
    band = ((y - 0.02) / 0.14) % 1.0
    f.material_index = 1 if 0.5 < band < 0.9 else 0
bm.to_mesh(abd.data); bm.free()

thorax = sphere("thorax", 0.245, (0, -0.06, 0.50), (1.0, 1.15, 0.88), (THORAX,), seg=32, rings=20, parent=root)
bm = bmesh.new(); bm.from_mesh(thorax.data)
for v in bm.verts:                                # humped back, flatter belly
    if v.co.z > 0.50: v.co.z = 0.50 + (v.co.z - 0.50) * 1.12
    else: v.co.z = 0.50 + (v.co.z - 0.50) * 0.8
bm.to_mesh(thorax.data); bm.free()
sphere("scutellum", 0.07, (0, 0.14, 0.60), (1.0, 0.9, 0.6), (THORAX,), parent=root)
# bristles on the thorax
hairs = []
for i in range(14):
    a = random.uniform(-0.9, 0.9); y = random.uniform(-0.2, 0.12)
    base = Vector((0.2 * math.sin(a), y, 0.50 + 0.2 * math.cos(a) * 0.95))
    tip = base + Vector((0.03 * math.sin(a), 0.04, 0.05 * math.cos(a) + 0.015))
    hairs.append(tube(f"hair{i}", base, tip, 0.006, 0.001, (HAIR,), verts=5))
finish(join(hairs, "bristles"), "bristles", (HAIR,), smooth=True, subdiv=0, parent=root)

# head, faceted compound eyes, antennae, proboscis
head = sphere("head", 0.165, (0, -0.35, 0.55), (1.12, 0.9, 0.98), (CHITIN,), parent=root)
for sx, nm in ((-1, "eyeL"), (1, "eyeR")):
    icosphere(nm, 0.115, (sx * 0.115, -0.40, 0.575), (0.82, 1.0, 1.12), (EYE,), sub=3, parent=head)
    base = Vector((sx * 0.05, -0.47, 0.64)); mid = base + Vector((sx * 0.06, -0.10, 0.12)); tip = mid + Vector((sx * 0.06, -0.12, 0.02))
    a1 = tube(f"ant{nm}", base, mid, 0.012, 0.008, (LEG,), verts=6); a2 = tube(f"ant2{nm}", mid, tip, 0.008, 0.003, (LEG,), verts=6)
    finish(join([a1, a2], f"antenna{nm[-1]}"), f"antenna{nm[-1]}", (LEG,), smooth=True, subdiv=0, parent=head)
prob = tube("proboscis", (0, -0.42, 0.47), (0, -0.50, 0.36), 0.035, 0.02, (LEG,), verts=8, parent=head)
sphere("labellum", 0.035, (0, -0.50, 0.355), (1.3, 1.0, 0.6), (LEG,), parent=head)

# six legs: femur (origin at hip) -> tibia (origin at knee) -> tarsus tip
ROWS = ((-0.16, -0.16), (0.0, 0.02), (0.16, 0.20))    # (hip y, knee y offset) front -> back
for sx, S in ((-1, "L"), (1, "R")):
    for row, (hy, dy) in enumerate(ROWS):
        hip = Vector((sx * 0.17, hy, 0.40)); knee = Vector((sx * 0.33, hy + dy, 0.50)); foot = Vector((sx * 0.43, hy + dy * 2.0, 0.0))
        femur = tube(f"leg{S}{row}", hip, knee, 0.03, 0.024, (LEG,), verts=8, parent=root)
        sphere(f"knee{S}{row}", 0.028, knee, mats=(LEG,), seg=10, rings=8, subdiv=0, parent=femur)
        tibia = tube(f"leg{S}{row}_tibia", knee, foot, 0.02, 0.012, (LEG,), verts=8, parent=femur)
        sphere(f"tarsus{S}{row}", 0.018, foot + Vector((sx * 0.02, 0, 0.01)), (1.3, 1.0, 0.7), (LEG,), seg=8, rings=6, subdiv=0, parent=tibia)

# wings: flat membrane outline + vein tubes, origin at the root, spanning +X (mirrored for the left)
def wing(name, sx):
    L, W = 0.66, 0.30
    top, bot = [], []
    for i in range(1, 24):
        t = i / 24
        top.append((L * t, W * 0.55 * math.sqrt(math.sin(math.pi * t)) * (1 - 0.25 * t)))
        bot.append((L * t, -W * 0.45 * math.sin(math.pi * t) ** 0.8 * (1 + 0.1 * t)))
    outline = [(0.0, 0.0)] + top + [(L, 0.0)] + bot[::-1]
    mesh = bpy.data.meshes.new(name); verts = [(sx * x, -y * 0.35 + x * 0.0, y) for x, y in outline]
    # lie flat: local plane is X (span) x Y (chord); z = 0. Remap: chord along -Y (backward) so the wing trails.
    verts = [(sx * x, y * -1.0 + x * 0.05, 0.0) for x, y in outline]
    mesh.from_pydata(verts, [], [list(range(len(verts)))]); mesh.update()
    o = bpy.data.objects.new(name, mesh); bpy.context.collection.objects.link(o)
    bpy.ops.object.select_all(action="DESELECT"); o.select_set(True); bpy.context.view_layer.objects.active = o
    bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_all(action="SELECT"); bpy.ops.mesh.quads_convert_to_tris(); bpy.ops.object.mode_set(mode="OBJECT")
    veins = []
    for j, (ex, ey) in enumerate([(L, 0.0), (L * 0.85, W * 0.42), (L * 0.6, W * 0.5), (L * 0.9, -W * 0.3), (L * 0.55, -W * 0.42), (L * 0.3, W * 0.45)]):
        a = Vector((0, 0, 0.002)); b = Vector((sx * ex, -ey + ex * 0.05, 0.002))
        veins.append(tube(f"{name}vein{j}", a, b, 0.006, 0.003, (VEIN,), verts=4))
    cross = tube(f"{name}cross", Vector((sx * L * 0.4, W * 0.4 + L * 0.02, 0.002)), Vector((sx * L * 0.45, -W * 0.3 + L * 0.02, 0.002)), 0.004, 0.004, (VEIN,), verts=4)
    o.data.materials.append(MEMBRANE)
    w = join([o] + veins + [cross], name)
    w.location = (sx * 0.07, -0.02, 0.665)
    w.parent = root; w.matrix_parent_inverse = root.matrix_world.inverted()
    return w
wing("wingL", -1); wing("wingR", 1)
for sx, nm in ((-1, "haltereL"), (1, "haltereR")):
    sphere(nm, 0.02, (sx * 0.2, 0.12, 0.55), mats=(CHITIN,), seg=8, rings=6, subdiv=0, parent=root)

# ----------------------------------------------------------------------------- export
bpy.ops.object.select_all(action="SELECT")
bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB", export_apply=True, export_yup=True,
                          export_animations=False, export_skins=False, export_cameras=False, export_lights=False)
print("EXPORTED", OUT, "objects:", len(bpy.data.objects))

# optional preview render (EEVEE, three-quarter view)
if PREVIEW:
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = 900, 600
    bpy.ops.object.camera_add(location=(1.3, -1.6, 1.15)); cam = bpy.context.active_object
    cam.rotation_euler = (Vector((0, 0, 0.45)) - cam.location).to_track_quat("-Z", "Y").to_euler(); scene.camera = cam
    bpy.ops.object.light_add(type="SUN", location=(2, -2, 4)); sun = bpy.context.active_object; sun.data.energy = 4
    sun.rotation_euler = (Vector((0, 0, 0)) - sun.location).to_track_quat("-Z", "Y").to_euler()
    bpy.ops.object.light_add(type="AREA", location=(-2, -1, 2)); fill = bpy.context.active_object; fill.data.energy = 200; fill.data.size = 3
    fill.rotation_euler = (Vector((0, 0, 0.4)) - fill.location).to_track_quat("-Z", "Y").to_euler()
    world = bpy.data.worlds.new("w"); scene.world = world; world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.75, 0.72, 0.62, 1); world.node_tree.nodes["Background"].inputs[1].default_value = 0.8
    bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, 0)); floor = bpy.context.active_object; floor.data.materials.append(material("floor", (0.35, 0.28, 0.18), 0.9))
    scene.render.filepath = PREVIEW; bpy.ops.render.render(write_still=True); print("PREVIEW", PREVIEW)

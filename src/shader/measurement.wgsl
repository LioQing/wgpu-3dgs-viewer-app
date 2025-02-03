// Vertex

const line_width_factor: f32 = 0.01;

struct HitPair {
    hit_0: vec3<f32>,
    color: u32,
    hit_1: vec3<f32>,
    line_width: f32,
}
@group(0) @binding(0)
var<storage, read> hit_pairs: array<HitPair>;

struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    size: vec2<f32>,
}
@group(0) @binding(1)
var<uniform> camera: Camera;

@vertex
fn vert_main(
    @builtin(vertex_index) vert_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> FragmentInput {
    var out: FragmentInput;

    let hit_pair = hit_pairs[instance_index];

    out.color = unpack4x8unorm(hit_pair.color);

    let pos_view_pair = array<vec4<f32>, 2>(
        camera.view * vec4<f32>(hit_pair.hit_0, 1.0),
        camera.view * vec4<f32>(hit_pair.hit_1, 1.0),
    );
    let pos_proj_pair = array<vec4<f32>, 2>(
        camera.proj * pos_view_pair[0],
        camera.proj * pos_view_pair[1],
    );

    let dir = normalize(
        (pos_proj_pair[0].xy / pos_proj_pair[0].w - pos_proj_pair[1].xy / pos_proj_pair[1].w)
        * camera.size
    );

    let normal = vec2<f32>(dir.y, -dir.x);
    let normal_dir = f32(vert_index % 2u) * 2.0 - 1.0;
    let normal_offset = normal * normal_dir * hit_pair.line_width * line_width_factor;

    let pair_index = vert_index < 2u || vert_index == 3u;
    let pos_view = pos_view_pair[u32(pair_index)];
    let pos_proj = pos_proj_pair[u32(pair_index)];
    let aspect_ratio = camera.size.y / camera.size.x;

    let parallel_dir = -(f32(pair_index) * 2.0 - 1.0);
    let parallel_offset = dir * parallel_dir * hit_pair.line_width * line_width_factor;
    let total_offset = normal_offset + parallel_offset;

    out.clip_pos = pos_proj
        + vec4<f32>(
            total_offset * pos_proj.w * vec2<f32>(aspect_ratio, 1.0) / length(pos_view.xyz),
            0.0, 0.0,
        );

    return out;
}

// Fragment

struct FragmentInput {
    @location(0) color: vec4<f32>,

    @builtin(position) clip_pos: vec4<f32>,
}

@fragment
fn frag_main(in: FragmentInput) -> @location(0) vec4<f32> {
    return in.color;
}
from helper_classes import *
import matplotlib.pyplot as plt


def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            # screen is on origin
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)
            ray = Ray(origin, direction)
            color = np.zeros(3)

            # This is the main loop where each pixel color is computed.
            # TODO
            scene = Scene(camera, ambient, lights, objects, screen_size)
            intersection = find_intersection(ray, objects)
            if intersection is not None:
                hit_point, obj = intersection
                hit_point += get_point_bias(obj, hit_point)
                color = get_color(
                    scene, obj, ray, hit_point, 0, max_depth)

            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color, 0, 1)

    return image


def find_intersection(ray, objects):
    t, object = ray.nearest_intersected_object(objects)
    if object is None:
        return None
    hit_point = ray.origin + ray.direction * t
    return hit_point, object


def get_color(scene, closest_object, ray, intersection_point, depth, max_depth):
    color = np.zeros(3)
    color += calc_ambient_color(scene.ambient, closest_object)

    for light in scene.lights:
        shadow_coefficient = calc_shadow_coefficient(
            light, intersection_point, scene.objects)

        if shadow_coefficient == 0:
            continue
        color += calc_diffuse_color(intersection_point, light, closest_object)
        color += calc_specular_color(scene,
                                     intersection_point, light, closest_object)

    depth += 1
    if depth < max_depth:
        reflected_ray = construct_reflective_ray(
            intersection_point, ray, closest_object)
        intersection = find_intersection(reflected_ray, scene.objects)
        if intersection is not None:
            hit_point, obj = intersection
            hit_point += get_point_bias(obj, hit_point)
            reflected_color = get_color(
                scene, obj, reflected_ray, hit_point, depth, max_depth)
            color += closest_object.reflection * reflected_color
    return color


def calc_ambient_color(ambient, object):
    return ambient * object.ambient


def calc_diffuse_color(hit_point, light, object):
    N = calc_object_norm(object, hit_point)
    L_direction = light.get_light_ray(hit_point).direction

    return light.get_intensity(hit_point) * object.diffuse * np.dot(N, L_direction)


def calc_specular_color(scene, hit_point, light, object):
    V_hat = normalize(scene.camera - hit_point)
    L_direction = -1 * light.get_light_ray(hit_point).direction
    R_hat = reflected(L_direction, calc_object_norm(object, hit_point))

    return light.get_intensity(hit_point) * object.specular * (np.dot(V_hat, R_hat) ** object.shininess)


def construct_reflective_ray(hit_point, ray, object):
    return Ray(hit_point, reflected(ray.direction, calc_object_norm(object, hit_point)))


def calc_shadow_coefficient(light, hit_point, objects):
    ray = light.get_light_ray(hit_point)
    distance, object = ray.nearest_intersected_object(objects)

    if object is None:
        return 1

    if distance < light.get_distance_from_light(hit_point):
        return 0

    return 1


def get_point_bias(object, hit_point):
    return 0.01 * object.compute_normal(hit_point)


def calc_object_norm(object, hit_point):
    return object.compute_normal(hit_point)


def your_own_scene():
    camera = np.array([0, 0, 1])
    lights = []
    objects = []
    return camera, lights, objects


class Scene:
    def __init__(self, camera, ambient, lights, objects, screen_size):
        self.camera = camera
        self.ambient = ambient
        self.lights = lights
        self.objects = objects
        self.screen_size = screen_size

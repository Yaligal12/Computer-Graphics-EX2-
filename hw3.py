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
                hit_point, obj = find_intersection(ray, objects)
                hit_point += get_point_bias(obj, hit_point)
                color = get_color(
                    scene, obj, ray, hit_point, 0, max_depth)
            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color, 0, 1)

    return image


def find_intersection(ray, objects):
    intersection = ray.nearest_intersected_object(objects)

    if intersection is None:
        return None

    t = intersection[0]
    hit_point = ray.origin + ray.direction * t
    return hit_point, intersection[1]


def get_color(scene, closest_object, ray, intersection_point, depth, max_depth):
    pass


def calc_ambient_color(ambient, obj):
    pass


def calc_diffuse_color(scene, hit_point, ray, light):
    pass


def calc_specular_color(scene, hit_point, ray, light):
    pass


def construct_reflective_ray(hit_point, ray, object):
    return Ray(hit_point, reflected(ray.direction, calc_object_norm(object, hit_point)))


def calc_shadow_coefficient(light, hit_point, objects):
    pass


def get_point_bias(object, point):
    return 0.01 * calc_object_norm(object, point)


def calc_object_norm(object, point):

    if isinstance(object, Plane):
        return object.normal
    elif isinstance(object, Triangle):
        pass
    elif isinstance(object, Pyramid):
        pass
    elif isinstance(object, Sphere):
        pass
    else:
        raise ValueError("Object type not supported")


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

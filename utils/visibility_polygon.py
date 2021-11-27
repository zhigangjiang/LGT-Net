"""
@date: 2021/7/20
@description: reference https://www.redblobgames.com/articles/visibility/
"""
import math
import numpy as np
from functools import cmp_to_key as ctk
from PIL import Image


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class EndPoint(Point):
    def __init__(self, x: float, y: float, begins_segment: bool = None, segment=None, angle: float = None):
        super().__init__(x, y)
        self.begins_segment = begins_segment
        self.segment = segment
        self.angle = angle


class Segment:
    def __init__(self, x1: float, y1: float, x2: float, y2: float, d: float = None):
        self.p1 = EndPoint(x1, y1)
        self.p2 = EndPoint(x2, y2)
        self.p1.segment = self
        self.p2.segment = self
        self.d = d


def calculate_end_point_angles(light_source: Point, segment: Segment) -> None:
    x = light_source.x
    y = light_source.y
    dx = 0.5 * (segment.p1.x + segment.p2.x) - x
    dy = 0.5 * (segment.p1.y + segment.p2.y) - y
    segment.d = (dx * dx) + (dy * dy)
    segment.p1.angle = math.atan2(segment.p1.y - y, segment.p1.x - x)
    segment.p2.angle = math.atan2(segment.p2.y - y, segment.p2.x - x)


def set_segment_beginning(segment: Segment) -> None:
    d_angle = segment.p2.angle - segment.p1.angle
    if d_angle <= -math.pi:
        d_angle += 2 * math.pi
    if d_angle > math.pi:
        d_angle -= 2 * math.pi
    segment.p1.begins_segment = d_angle > 0
    segment.p2.begins_segment = not segment.p1.begins_segment


def endpoint_compare(point_a: EndPoint, point_b: EndPoint):
    if point_a.angle > point_b.angle:
        return 1
    if point_a.angle < point_b.angle:
        return -1
    if not point_a.begins_segment and point_b.begins_segment:
        return 1
    if point_a.begins_segment and not point_b.begins_segment:
        return -1
    return 0


def polygon_to_segments(polygon: np.array) -> np.array:
    segments = []
    polygon = np.concatenate((polygon, [polygon[0]]))
    for i in range(len(polygon) - 1):
        p1 = polygon[i]
        p2 = polygon[i + 1]
        segments.append([p1, p2])
    segments = np.array(segments)
    return segments


def segment_in_front_of(segment_a: Segment, segment_b: Segment, relative_point: Point):
    def left_of(segment: Segment, point: Point):
        cross = (segment.p2.x - segment.p1.x) * (point.y - segment.p1.y) - (segment.p2.y - segment.p1.y) * (
                point.x - segment.p1.x)
        return cross < 0

    def interpolate(point_a: Point, point_b: Point, f: float):
        point = Point(x=point_a.x * (1 - f) + point_b.x * f,
                      y=point_a.y * (1 - f) + point_b.y * f)
        return point

    a1 = left_of(segment_a, interpolate(segment_b.p1, segment_b.p2, 0.01))
    a2 = left_of(segment_a, interpolate(segment_b.p2, segment_b.p1, 0.01))
    a3 = left_of(segment_a, relative_point)
    b1 = left_of(segment_b, interpolate(segment_a.p1, segment_a.p2, 0.01))
    b2 = left_of(segment_b, interpolate(segment_a.p2, segment_a.p1, 0.01))
    b3 = left_of(segment_b, relative_point)
    if b1 == b2 and not (b2 == b3):
        return True
    if a1 == a2 and a2 == a3:
        return True
    if a1 == a2 and not (a2 == a3):
        return False
    if b1 == b2 and b2 == b3:
        return False
    return False


def line_intersection(point1: Point, point2: Point, point3: Point, point4: Point):
    a = (point4.y - point3.y) * (point2.x - point1.x) - (point4.x - point3.x) * (point2.y - point1.y)
    b = (point4.x - point3.x) * (point1.y - point3.y) - (point4.y - point3.y) * (point1.x - point3.x)
    assert a != 0 or a == b, "center on polygon, it not support!"
    if a == 0:
        s = 1
    else:
        s = b / a

    return Point(
        point1.x + s * (point2.x - point1.x),
        point1.y + s * (point2.y - point1.y)
    )


def get_triangle_points(origin: Point, angle1: float, angle2: float, segment: Segment):
    p1 = origin
    p2 = Point(origin.x + math.cos(angle1), origin.y + math.sin(angle1))
    p3 = Point(0, 0)
    p4 = Point(0, 0)

    if segment:
        p3.x = segment.p1.x
        p3.y = segment.p1.y
        p4.x = segment.p2.x
        p4.y = segment.p2.y
    else:
        p3.x = origin.x + math.cos(angle1) * 2000
        p3.y = origin.y + math.sin(angle1) * 2000
        p4.x = origin.x + math.cos(angle2) * 2000
        p4.y = origin.y + math.sin(angle2) * 2000

    #  use the endpoint directly when the rays are parallel to segment
    if abs(segment.p1.angle - segment.p2.angle) < 1e-6:
        return [p4, p3]

    # it's maybe generate error coordinate when the rays are parallel to segment
    p_begin = line_intersection(p3, p4, p1, p2)
    p2.x = origin.x + math.cos(angle2)
    p2.y = origin.y + math.sin(angle2)
    p_end = line_intersection(p3, p4, p1, p2)

    return [p_begin, p_end]


def calc_visible_polygon(center: np.array, polygon: np.array = None, segments: np.array = None, show: bool = False):
    if segments is None and polygon is not None:
        segments = polygon_to_segments(polygon)

    origin = Point(x=center[0], y=center[1])
    endpoints = []
    for s in segments:
        p1 = s[0]
        p2 = s[1]
        segment = Segment(x1=p1[0], y1=p1[1], x2=p2[0], y2=p2[1])
        calculate_end_point_angles(origin, segment)
        set_segment_beginning(segment)
        endpoints.extend([segment.p1, segment.p2])

    open_segments = []
    output = []
    begin_angle = 0
    endpoints = sorted(endpoints, key=ctk(endpoint_compare))

    for pas in range(2):
        for endpoint in endpoints:
            open_segment = open_segments[0] if len(open_segments) else None
            if endpoint.begins_segment:
                index = 0
                segment = open_segments[index] if index < len(open_segments) else None
                while segment and segment_in_front_of(endpoint.segment, segment, origin):
                    index += 1
                    segment = open_segments[index] if index < len(open_segments) else None

                if not segment:
                    open_segments.append(endpoint.segment)
                else:
                    open_segments.insert(index, endpoint.segment)
            else:
                if endpoint.segment in open_segments:
                    open_segments.remove(endpoint.segment)

            if open_segment is not (open_segments[0] if len(open_segments) else None):
                if pas == 1 and open_segment:
                    triangle_points = get_triangle_points(origin, begin_angle, endpoint.angle, open_segment)
                    output.extend(triangle_points)
                begin_angle = endpoint.angle

    output_polygon = []
    # Remove duplicate
    for i, p in enumerate(output):
        q = output[(i + 1) % len(output)]
        if int(p.x * 10000) == int(q.x * 10000) and int(p.y * 10000) == int(q.y * 10000):
            continue
        output_polygon.append([p.x, p.y])

    output_polygon.reverse()
    output_polygon = np.array(output_polygon)

    if show:
        visualization(segments, output_polygon, center)
    return output_polygon


def visualization(segments: np.array, output_polygon: np.array, center: np.array, side_l=1000):
    """
    :param segments: original segments
    :param output_polygon: result polygon
    :param center: visibility center
    :param side_l: side length of board
    :return:
    """
    try:
        import cv2
        import matplotlib.pyplot as plt
    except ImportError:
        print("visualization need cv2 and matplotlib")
        return
    offset = np.array([side_l / 2, side_l / 2]) - center
    segments = segments + offset
    output_polygon = output_polygon + offset
    origin = np.array([side_l / 2, side_l / 2])

    # +0.5 as board
    scale = side_l / 2.5 / np.abs(segments - origin).max()
    board = np.zeros((side_l, side_l))
    for segment in segments:
        segment = (segment - origin) * scale + origin
        segment = segment.astype(np.int)
        cv2.line(board, tuple(segment[0]), tuple(segment[1]), 0.5, thickness=3)
    board = cv2.drawMarker(board, tuple(origin.astype(np.int)), 1, thickness=3)

    output_polygon = (output_polygon - origin) * scale + origin
    board = cv2.drawContours(board, [output_polygon.astype(np.int)], 0, 1, 3)
    board = cv2.drawMarker(board, tuple(origin.astype(np.int)), 1, thickness=3)
    plt.axis('off')
    plt.imshow(board)
    plt.show()


if __name__ == '__main__':
    import numpy as np

    from dataset.mp3d_dataset import MP3DDataset
    from utils.boundary import depth2boundaries
    from utils.conversion import uv2xyz, depth2xyz
    from visualization.boundary import draw_boundaries
    from visualization.floorplan import draw_floorplan, draw_iou_floorplan

    mp3d_dataset = MP3DDataset(root_dir='../src/dataset/mp3d', mode='train',
                               split_list=[['e9zR4mvMWw7', '2224be23a70a475ea6daa55d4c90a91b']])
    gt = mp3d_dataset.__getitem__(0)
    gt['corners'] = gt['corners'][gt['corners'][..., 0] + gt['corners'][..., 1] != 0]  # Take effective corners

    img = draw_floorplan(depth2xyz(gt['depth'])[:, ::2], fill_color=[1, 1, 1, 0],
                         show=True, scale=1, marker_color=[0, 0, 1, 1], side_l=1024)
    # img = draw_iou_floorplan(gt_xz=uv2xyz(gt['corners'])[..., ::2],
    #                          dt_xz=calc_visible_polygon(np.array([0, 0]), uv2xyz(gt['corners'])[..., ::2]),
    #                          dt_board_color=[0, 0, 1, 0],
    #                          gt_board_color=[0, 0, 1, 0],
    #                          show=True, side_l=1024)

    result = Image.fromarray((img[250: -100, 100:-20] * 255).astype(np.uint8))
    result.save('../src/fig/sample3.png')

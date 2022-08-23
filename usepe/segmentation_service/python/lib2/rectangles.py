""" Decomposition of rectilinear polygon into rectangles
    see: https://stackoverflow.com/a/6634668
    ported from: https://github.com/mikolalysenko/rectangle-decomposition """

import math
from functools import cmp_to_key

import networkx as nx
import numpy as np
from intervaltree import IntervalTree
from shapely.geometry import MultiPolygon, Polygon, box, mapping

_INF = float("inf")


class Vertex:
    def __init__(self, point, path, index, concave):
        self.point = point
        self.path = path
        self.index = index
        self.concave = concave
        self.next = None
        self.prev = None
        self.visited = False

    def copy(self):
        c = Vertex(self.point, self.path, self.index, self.concave)
        c.next = self.next
        c.prev = self.prev
        c.visits = self.visits
        return c


class Segment:
    def __init__(self, start, end, direction):
        a = start.point[direction ^ 1]
        b = end.point[direction ^ 1]
        if a < b:
            self.val = [a, b]
        else:
            self.val = [b, a]
        self.start = start
        self.end = end
        self.direction = direction
        self.number = -1

    def __getitem__(self, idx):
        return self.val[idx]


def _test_segment(a, b, tree, direction):
    ax = a.point[direction ^ 1]
    bx = b.point[direction ^ 1]

    segments = tree.at(a.point[direction])
    result = False
    for s in segments:
        x = s.data.start.point[direction ^ 1]
        result = result or (ax < x and x < bx)
    return result


def _test_overlap(a, b, tree, direction):
    ax = a.point[direction]
    bx = b.point[direction]

    segments = tree.overlap(a.point[direction ^ 1], b.point[direction ^ 1])
    result = False
    for s in segments:
        x = s.data.start.point[direction]
        result = result or (ax == x) or (bx == x)
    return result


def _get_diagonals(vertices, paths, direction, tree, c_tree):
    concave = [v for v in vertices if v.concave]

    def concave_sort_function(a, b):
        d = a.point[direction] - b.point[direction]
        if d:
            return d
        return a.point[direction ^ 1] - b.point[direction ^ 1]

    concave.sort(key=cmp_to_key(concave_sort_function))

    diagonals = []
    for i in range(1, len(concave)):
        a = concave[i - 1]
        b = concave[i]
        if a.point[direction] == b.point[direction]:
            if a.path == b.path:
                n = len(paths[a.path])
                d = (a.index - b.index + n) % n
                if d == 1 or d == n - 1:
                    continue

            # if from the same path and not consecutive points -> check if other segment of the path overlaps the diagonal [a,b]
            ## MISSING wrong diagonals when polygons touching in one vertex
            # tree_parallel = IntervalTree()
            # pa = paths[a.path]
            # pb = paths[b.path]
            # if a.path == b.path:
            #     for j in range(0, len(pa)):
            #         a_parallel = pa[j]
            #         b_parallel = pa[(j + 1) % len(pa)]
            #         if a_parallel.point[direction] == b_parallel.point[direction]:
            #             seg = Segment(a_parallel, b_parallel, direction)
            #             tree_parallel[seg[0] : seg[1]] = seg
            # else:
            #     for j in range(0, len(pa)):
            #         a_parallel = pa[j]
            #         an_parallel = pa[(j + 1) % len(pa)]
            #         if a_parallel.point[direction] == an_parallel.point[direction]:
            #             seg = Segment(a_parallel, an_parallel, direction)
            #             tree_parallel[seg[0] : seg[1]] = seg

            #     for j in range(0, len(pb)):
            #         b_parallel = pb[j]
            #         bn_parallel = pb[(j + 1) % len(pb)]
            #         if b_parallel.point[direction] == bn_parallel.point[direction]:
            #             seg = Segment(b_parallel, bn_parallel, direction)
            #             tree_parallel[seg[0] : seg[1]] = seg

            if _test_overlap(a, b, c_tree, direction):
                continue

            if not _test_segment(a, b, tree, direction):
                diagonals.append(Segment(a, b, direction))

    return diagonals


def _find_crossings(hdiagonals, vdiagonals):
    htree = IntervalTree()
    for seg in hdiagonals:
        htree[seg[0] - 1e-9 : seg[1] + 1e-9] = seg
    crossings = []
    for v in vdiagonals:
        x = v.start.point[0]
        segments = htree.at(v.start.point[1])
        for h in segments:
            x = h.data.start.point[0]
            if v[0] <= x and x <= v[1]:
                crossings.append([h.data, v])
    return crossings


def _find_splitters(hdiagonals, vdiagonals):
    if not hdiagonals and not vdiagonals:
        return []
    crossings = _find_crossings(hdiagonals, vdiagonals)

    for i in range(0, len(hdiagonals)):
        hdiagonals[i].number = i
    for i in range(0, len(vdiagonals)):
        vdiagonals[i].number = i + len(hdiagonals)
    if not crossings:
        return hdiagonals + vdiagonals

    graph = nx.Graph()
    graph.add_nodes_from([d.number for d in hdiagonals], bipartite=0)
    graph.add_nodes_from([d.number for d in vdiagonals], bipartite=1)
    graph.add_edges_from([[c[0].number, c[1].number] for c in crossings])
    isolates = list(nx.isolates(graph))
    graph.remove_nodes_from(isolates)
    topn = [n for n in graph.nodes if graph.nodes[n]["bipartite"] == 0]
    matching = nx.bipartite.maximum_matching(graph, top_nodes=topn)
    graph.add_nodes_from(isolates)
    vertex_cover = nx.bipartite.to_vertex_cover(graph, matching, top_nodes=topn)
    # selected = set(graph) - vertex_cover | set(isolates)
    selected = set(graph) - vertex_cover

    diag = hdiagonals + vdiagonals
    return [diag[s] for s in selected]


def _split_segment(segment):
    a = segment.start
    b = segment.end

    pa = a.prev
    na = a.next
    pb = b.prev
    nb = b.next

    ao = pa.point[segment.direction] == a.point[segment.direction]
    bo = pb.point[segment.direction] == b.point[segment.direction]

    if ao and bo:
        a.prev = pb
        pb.next = a
        b.prev = pa
        pa.next = b

    elif ao and not bo:
        a.prev = b
        b.next = a
        pa.next = nb
        nb.prev = pa

    elif not ao and bo:
        a.next = b
        b.prev = a
        na.prev = pb
        pb.next = na

    elif not ao and not bo:
        a.next = nb
        nb.prev = a
        b.next = na
        na.prev = b

    a.concave = False
    b.concave = False

    segment.start = a
    segment.end = b
    return segment


def _split_concave(vertices):
    # First step: build segment tree from vertical segments
    lefttree = IntervalTree()
    righttree = IntervalTree()
    for v in vertices:
        if v.next.point[1] == v.point[1]:
            seg = Segment(v, v.next, 1)
            if v.next.point[0] < v.point[0]:
                lefttree[seg[0] : seg[1]] = seg
            else:
                righttree[seg[0] : seg[1]] = seg
    ii = 0
    for v in vertices:
        if not v.concave:
            continue
        # Compute orientation
        y = v.point[1]
        if v.prev.point[0] == v.point[0]:
            direction = v.prev.point[1] < y
        else:
            direction = v.next.point[1] < y
        direction = 1 if direction else -1

        # Scan a horizontal ray
        closestSegment = None
        closestDistance = _INF * direction
        if direction < 0:
            for h in righttree.at(v.point[0]):
                x = h.data.start.point[1]
                if x < y and x > closestDistance:
                    closestDistance = x
                    closestSegment = h
        else:
            for h in lefttree.at(v.point[0]):
                x = h.data.start.point[1]
                if x > y and x < closestDistance:
                    closestDistance = x
                    closestSegment = h

        # Create two splitting vertices
        splitA = Vertex([v.point[0], closestDistance], -1, ii, False)
        splitB = Vertex([v.point[0], closestDistance], -2, ii, False)
        v.concave = False
        # Split vertices
        splitA.prev = closestSegment.data.start
        closestSegment.data.start.next = splitA
        splitB.next = closestSegment.data.end
        closestSegment.data.end.prev = splitB
        # Update segment tree
        if direction < 0:
            tree = righttree
        else:
            tree = lefttree

        tree.remove(closestSegment)
        seg = Segment(closestSegment.data.start, splitA, 1)
        tree.addi(seg[0], seg[1], seg)
        seg = Segment(splitB, closestSegment.data.end, 1)
        tree.addi(seg[0], seg[1], seg)

        # Cut v, 2 different cases
        if v.prev.point[0] == v.point[0]:
            splitA.next = v
            splitB.prev = v.prev

        else:
            splitA.next = v.next
            splitB.prev = v

        # Fix up links
        splitA.next.prev = splitA
        splitB.prev.next = splitB

        # Append vertices
        vertices.append(splitA)
        vertices.append(splitB)

        ii += 1

    return vertices


def _find_regions(vertices):
    n = len(vertices)
    import matplotlib.pyplot as plt

    # plt.figure()
    plt.show()
    for i in range(0, n):
        vertices[i].visited = False

    rectangles = []
    for v in vertices:
        if v.visited:
            continue

        lo = [_INF, _INF]
        hi = [-_INF, -_INF]

        while not v.visited:
            for j in range(0, 2):
                lo[j] = min(v.point[j], lo[j])
                hi[j] = max(v.point[j], hi[j])
            v.visited = True
            # plt.plot(v.point[0], v.point[1], "x", color="red")
            v = v.next
        rectangles.append([lo, hi])

    return rectangles


def decompose_region(paths, clockwise):
    """Find the minimum rectangle cover for orthogonal polygons"""
    vertices = []
    npaths = []
    for i in range(0, len(paths)):
        path = paths[i]

        # Loop over all rings present in the polygon and find out
        # which vertices are concave
        n = len(path)
        prev, cur, next = path[n - 3 : n]
        npaths.append([])
        for j in range(0, n):
            prev, cur, next = cur, next, path[j]

            concave = False
            if prev[0] == cur[0]:
                if next[0] == cur[0]:
                    continue
                dir0 = prev[1] < cur[1]
                dir1 = cur[0] < next[0]
                concave = dir0 == dir1
            else:
                if next[1] == cur[1]:
                    continue
                dir0 = prev[0] < cur[0]
                dir1 = cur[1] < next[1]
                concave = dir0 != dir1

            if clockwise:
                concave = not concave

            vtx = Vertex(cur, i, (j + n - 1) % n, concave)
            npaths[i].append(vtx)
            vertices.append(vtx)

    # Build up interval trees for all vertical and horizontal segments,
    # by looking at adjacent vertices that share the same x or y components
    htree = IntervalTree()
    vtree = IntervalTree()
    for p in npaths:
        for j in range(0, len(p)):
            a = p[j]
            b = p[(j + 1) % len(p)]
            if a.point[0] == b.point[0]:
                seg = Segment(a, b, 0)
                htree[seg[0] : seg[1]] = seg
            else:
                seg = Segment(a, b, 1)
                vtree[seg[0] : seg[1]] = seg

            if clockwise:
                a.prev = b
                b.next = a
            else:
                a.next = b
                b.prev = a

    # Find diagonals between concave vertices, horizontal diagonals are
    # drawn between vertical segments and vice versa
    hdiagonals = _get_diagonals(vertices, npaths, 0, vtree, htree)
    vdiagonals = _get_diagonals(vertices, npaths, 1, htree, vtree)

    # Create a bipartite graph where the nodes are the horizontal and vertical
    # diagonals and the edges are their intersections, the polygon will then
    # be split along as many of the diagonals that don't cross as possible
    # (i.e. the maximum indepent set of the graph)
    splitters = _find_splitters(hdiagonals, vdiagonals)
    splitters = [_split_segment(splitter) for splitter in splitters]

    # Finally, a cut is made from all remaining concave vertices
    vertices = _split_concave(vertices)
    ## plot all vertices to confirm all concave are selected
    # last point in all paths is removed since it is the same as the initial point

    # import matplotlib.pyplot as plt

    # plt.figure()
    # ax = plt.axes()
    # for pp in range(len(paths)):
    #     ax.plot(
    #         np.concatenate([np.array(paths[pp]), np.array(paths[pp])[0:1, :]])[:, 0],
    #         np.concatenate([np.array(paths[pp]), np.array(paths[pp])[0:1, :]])[:, 1],
    #         color="blue",
    #         linewidth=3,
    #     )
    #     ax.annotate(
    #         "",
    #         xy=(
    #             np.array(paths[pp])[-1, 0],
    #             np.array(paths[pp])[-1, 1],
    #         ),
    #         xycoords="data",
    #         xytext=(
    #             np.array(paths[pp])[-2, 0],
    #             np.array(paths[pp])[-2, 1],
    #         ),
    #         textcoords="data",
    #         arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
    #     )
    # for pp in range(len(vertices)):
    #     if vertices[pp].concave:
    #         ax.plot(
    #             vertices[pp].point[0],
    #             vertices[pp].point[1],
    #             marker="x",
    #             markersize=5,
    #             markeredgecolor="black",
    #         )
    # # for pp in range(len(hdiagonals)):
    # #     ax.plot(
    # #         [hdiagonals[pp].start.point[0], hdiagonals[pp].end.point[0]],
    # #         [hdiagonals[pp].start.point[1], hdiagonals[pp].end.point[1]],
    # #         linewidth=4,
    # #         color="red",
    # #         alpha=0.5,
    # #     )
    # # for pp in range(len(vdiagonals)):
    # #     ax.plot(
    # #         [vdiagonals[pp].start.point[0], vdiagonals[pp].end.point[0]],
    # #         [vdiagonals[pp].start.point[1], vdiagonals[pp].end.point[1]],
    # #         linewidth=4,
    # #         color="black",
    # #         alpha=0.5,
    # #     )
    # for pp in range(len(splitters)):
    #     ax.plot(
    #         [splitters[pp].start.point[0], splitters[pp].end.point[0]],
    #         [splitters[pp].start.point[1], splitters[pp].end.point[1]],
    #         linewidth=4,
    #         color="black",
    #         alpha=0.5,
    #     )
    # pathsOut = _find_regions(vertices)
    # for pp in range(len(pathsOut)):
    #     ax.plot(
    #         [
    #             pathsOut[pp][0][0],
    #             pathsOut[pp][0][0],
    #             pathsOut[pp][1][0],
    #             pathsOut[pp][1][0],
    #             pathsOut[pp][0][0],
    #         ],
    #         [
    #             pathsOut[pp][0][1],
    #             pathsOut[pp][1][1],
    #             pathsOut[pp][1][1],
    #             pathsOut[pp][0][1],
    #             pathsOut[pp][0][1],
    #         ],
    #         color="red",
    #         alpha=0.3,
    #     )
    # plt.show()

    return _find_regions(vertices)


def decompose(poly):
    # import matplotlib.pyplot as plt
    if math.isclose(poly.minimum_rotated_rectangle.area, poly.area):
        return MultiPolygon([poly])

    poly = mapping(poly)
    ## plot all polygons including the orientation
    # plt.figure()
    # ax = plt.axes()
    # for pp in range(len(poly["coordinates"])):
    #     ax.plot(np.array(poly["coordinates"][pp])[:, 0], np.array(poly["coordinates"][pp])[:, 1])
    #     ax.annotate(
    #         "",
    #         xy=(
    #             np.array(poly["coordinates"][pp])[-1, 0],
    #             np.array(poly["coordinates"][pp])[-1, 1],
    #         ),
    #         xycoords="data",
    #         xytext=(
    #             np.array(poly["coordinates"][pp])[-2, 0],
    #             np.array(poly["coordinates"][pp])[-2, 1],
    #         ),
    #         textcoords="data",
    #         arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
    #     )
    # plt.show()

    coords = [path[:-1] for path in reversed(poly["coordinates"])]
    decomp = decompose_region(coords, True)
    rects = [np.array(rect).flatten() for rect in decomp]
    return MultiPolygon([box(*rect) for rect in rects])

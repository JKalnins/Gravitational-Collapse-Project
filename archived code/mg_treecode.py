import numpy as np


class OctNode:
    """Stores the data for an octree node, and spawns its children if possible"""

    def __init__(self, center, size, masses, points, ids, leaves=[]):
        self.center = center  # center of the node's box
        self.size = size  # maximum side length of the box
        self.children = []  # start out assuming that the node has no children

        Npoints = len(points)

        if Npoints == 1:
            # if we're down to one point, we need to store stuff in the node
            leaves.append(self)
            self.COM = points[0]
            self.mass = masses[0]
            self.id = ids[0]
            self.g = np.zeros(3)  # at each point, we will want the gravitational field
        else:
            self.GenerateChildren(
                points, masses, ids, leaves
            )  # if we have at least 2 points in the node,
            # spawn its children

            """now we can sum the total mass and center of mass hierarchically,
            visiting each point once!"""
            com_total = np.zeros(3)  # running total for mass moments to get COM
            m_total = 0.0  # running total for masses
            for c in self.children:
                m, com = c.mass, c.COM
                m_total += m
                com_total += com * m  # add the moments of each child
            self.mass = m_total
            self.COM = com_total / self.mass

    def GenerateChildren(self, points, masses, ids, leaves):
        """Generates the node's children"""
        octant_index = (
            points > self.center
        )  # does all comparisons needed to determine points' octants
        for i in range(2):  # looping over the 8 octants
            for j in range(2):
                for k in range(2):
                    in_octant = np.all(octant_index == np.bool_([i, j, k]), axis=1)
                    if not np.any(in_octant):
                        continue  # if no particles, don't make a node
                    dx = (
                        0.5 * self.size * (np.array([i, j, k]) - 0.5)
                    )  # offset between parent and child box centers
                    self.children.append(
                        OctNode(
                            self.center + dx,
                            self.size / 2,
                            masses[in_octant],
                            points[in_octant],
                            ids[in_octant],
                            leaves,
                        )
                    )


def TreeWalk(node, node0, thetamax=0.7, G=1.0):
    """
    Adds the contribution to the field at node0's point due to particles in node.
    Calling this with the topnode as node will walk the tree to calculate the total
    field at node0.
    """
    dx = node.COM - node0.COM  # vector between nodes' centres of mass
    r = np.sqrt(np.sum(dx ** 2))  # distance between them
    if r > 0:
        # if the node only has one particle or theta is small enough,
        #  add the field contribution to value stored in node.g
        if (len(node.children) == 0) or (node.size / r < thetamax):
            node0.g += G * node.mass * dx / r ** 3
        else:
            # otherwise split up the node and repeat
            for c in node.children:
                TreeWalk(c, node0, thetamax, G)


def GravAccel(points, masses, thetamax=0.7, G=1.0):
    center = (
        np.max(points, axis=0) + np.min(points, axis=0)
    ) / 2  # center of bounding box
    topsize = np.max(
        np.max(points, axis=0) - np.min(points, axis=0)
    )  # size of bounding box
    leaves = []  # want to keep track of leaf nodes
    topnode = OctNode(
        center, topsize, masses, points, np.arange(len(masses)), leaves
    )  # build the tree

    accel = np.empty_like(points)
    for i, leaf in enumerate(leaves):
        TreeWalk(topnode, leaf, thetamax, G)  # do field summation
        accel[leaf.id] = leaf.g  # get the stored acceleration
    return accel

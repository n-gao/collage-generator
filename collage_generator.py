import argparse
import math
import random
from os import listdir
from os.path import exists, isfile, join

import cv2
import numpy as np


class Node:
    def __init__(
        self, N, alpha_t=None, parent=None, split=None, img=None, left=None, right=None
    ):
        self.parent = parent
        self.left = left
        self.right = right
        self.split = split
        self.img = img
        self.N = N
        self.alpha_t = alpha_t

    @property
    def depth(self):
        if self.parent == None:
            return 0
        else:
            return self.parent.depth + 1

    @property
    def alpha(self):
        if self.img is not None:
            return self.img.shape[1] / self.img.shape[0]
        alpha_left = self.left.alpha
        alpha_right = self.right.alpha
        if self.split == "V":
            return alpha_left + alpha_right
        else:
            return (alpha_left * alpha_right) / (alpha_left + alpha_right)


def load_images(folder):
    imgs = []
    for img in listdir(folder):
        try:
            img = cv2.imread(join(folder, img), cv2.IMREAD_COLOR)
            if img is None:
                raise IOError()
            imgs.append(img)
        except:
            pass
    return imgs


def sample_energies(energies, temperature=1):
    energies = np.array(energies)
    probs = np.exp(-energies / temperature)
    probs = probs / np.sum(probs)
    return np.random.choice(len(energies), p=probs)


def find_img_pair(alpha_t, L, temperature=1):
    p = 0
    q = len(L) - 1
    energies = []
    pairs = []
    while p < q:
        alpha_sum = L[p].alpha + L[q].alpha
        energies.append(abs(alpha_sum - alpha_t))
        pairs.append((p, q))
        if alpha_sum >= alpha_t:
            q = q - 1
        elif alpha_sum < alpha_t:
            p = p + 1
    i, j = pairs[sample_energies(energies, temperature)]
    return (i, j)


def generate_tree(L, node, temperature):
    if node.N == 1:
        energies = np.array([abs(l.alpha - node.alpha_t) for l in L])
        best_fit = L[sample_energies(energies, temperature)]
        node.img = best_fit.img
        L.remove(best_fit)
        return
    if node.N == 2:
        i, j = find_img_pair(node.alpha_t, L, temperature)
        node.left = L[i]
        node.right = L[j]
        node.left.parent = node
        node.right.paernt = node
        del L[j]
        del L[i]
        return
    node.split = np.random.choice(["V", "H"])
    if node.split == "V":
        node.left = Node(
            parent=node, N=math.floor(node.N / 2), alpha_t=node.alpha_t / 2
        )
        node.right = Node(
            parent=node, N=math.ceil(node.N / 2), alpha_t=node.alpha_t / 2
        )
    else:
        node.left = Node(
            parent=node, N=math.floor(node.N / 2), alpha_t=node.alpha_t * 2
        )
        node.right = Node(
            parent=node, N=math.ceil(node.N / 2), alpha_t=node.alpha_t * 2
        )
    generate_tree(L, node.left, temperature)
    generate_tree(L, node.right, temperature)


def adjust_tree(node, th):
    if node.img is not None:
        return
    if node.alpha > node.alpha_t * th:
        node.split = "H"
    elif node.alpha < node.alpha_t / th:
        node.split = "V"
    if node.split == "V":
        node.left.alpha_t = node.alpha_t / 2
        node.right.alpha_t = node.alpha_t / 2
    else:
        node.left.alpha_t = node.alpha_t * 2
        node.right.alpha_t = node.alpha_t * 2
    adjust_tree(node.left, th)
    adjust_tree(node.right, th)


def generate_and_adjust_tree(imgs, ratio, th, temperature):
    L = [Node(N=1, img=img) for img in imgs]
    L.sort(key=lambda node: node.alpha)
    root = Node(N=len(L), alpha_t=ratio)
    generate_tree(L, root, temperature)
    for i in range(10):
        adjust_tree(root, th)
    return root


def generate_best_tree(imgs, target_ratio, threshold=1e-4, temperature=1):
    best_tree = None
    best_error = -1
    for th in np.arange(10) / 50 + 0.55:
        for i in range(500):
            root = generate_and_adjust_tree(imgs, target_ratio, th, temperature)
            if abs(root.alpha - target_ratio) < best_error or best_error == -1:
                best_error = abs(root.alpha - target_ratio)
                best_tree = root
            if best_error < threshold:
                break
    return best_tree


def to_image(img, node, x, y, height):
    if node.img is not None:
        new_width = math.floor(node.alpha * height)
        res = cv2.resize(
            node.img, dsize=(new_width, height), interpolation=cv2.INTER_CUBIC
        )
        target_shape = img[y : height + y, x : new_width + x].shape
        img[y : height + y, x : new_width + x] = res[
            : target_shape[0], : target_shape[1]
        ]
        return
    alpha = node.alpha
    l_alpha = node.left.alpha
    r_alpha = node.right.alpha
    width = alpha * height
    if node.split == "V":
        to_image(img, node.left, x, y, height + 2)
        to_image(
            img, node.right, x + math.floor(width * l_alpha / alpha), y, height + 2
        )
    else:
        to_image(img, node.left, x, y, math.floor(height * alpha / l_alpha) + 2)
        to_image(
            img,
            node.right,
            x,
            y + math.floor(height * alpha / l_alpha),
            math.floor(height * alpha / r_alpha) + 2,
        )


def is_valid_folder(parser, arg):
    if not exists(arg) or isfile(arg):
        parser.error("The path %s does not exist or is a file!" % arg)
    else:
        return arg


def is_valid_file(parser, arg):
    if exists(arg) and not isfile(arg):
        parser.error("The %s is a folder!" % arg)
    else:
        return arg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colalge generator.")
    parser.add_argument(
        "src_folder",
        help="Source image folder",
        default=".",
        type=lambda x: is_valid_folder(parser, x),
    )
    parser.add_argument(
        "--output", default="collage.jpg", type=lambda x: is_valid_file(parser, x)
    )
    parser.add_argument("--width", default=4096, type=int)
    parser.add_argument("--height", default=2048, type=int)
    parser.add_argument("--temperature", default=1, type=float)
    args = parser.parse_args()

    print("Loading images...")
    imgs = load_images(args.src_folder)
    random.shuffle(imgs)
    target_ratio = args.width / args.height
    print("Generating tree...")
    tree = generate_best_tree(imgs, target_ratio, args.temperature)
    print("Building image...")
    img = np.ones((args.height, args.width, 3), dtype=np.uint8) * 255
    to_image(img, tree, 0, 0, args.height)
    print("Saving output...")
    cv2.imwrite(args.output, img)
    print("Done.")

import sys, os.path, cv2, numpy as np


def otsu(img: np.ndarray) -> np.ndarray:
    h = len(img)
    w = len(img[0])

    min = max = img[0][0]

    for i in range(h):
        for j in range(w):
            if min > img[i][j]:
                min = img[i][j]
            if max < img[i][j]:
                max = img[i][j]
    gisto = np.zeros(max - min + 1)
    for i in range(h):
        for j in range(w):
            gisto[img[i][j] - min] += 1
    sum, sumt = 0, 0
    for cur in range(max - min + 1):
        sum += gisto[cur]
        sumt += gisto[cur] * cur
    EDGE = 0
    sigm = -777
    al, be = 0, 0
    for cur in range(max - min):
        al += cur * gisto[cur]
        be += gisto[cur]

        w1 = be / sum

        a = (al / be) - ((sumt - al) / (sum - be))

        sigma = w1 * (1 - w1) * a * a

        if (sigma > sigm):
            sigm = sigma
            EDGE = cur
    Res = np.ones((h, w), dtype = int)

    for i in range(h):
        for j in range(w):
            if img[i][j] < EDGE:
                Res[i][j] = 0
            else:
                Res[i][j] = 255
    return Res
    pass  # insert your code here


def main():
    assert len(sys.argv) == 3
    src_path, dst_path = sys.argv[1], sys.argv[2]

    assert os.path.exists(src_path)
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    result = otsu(img)
    cv2.imwrite(dst_path, result)


if __name__ == '__main__':
    main()
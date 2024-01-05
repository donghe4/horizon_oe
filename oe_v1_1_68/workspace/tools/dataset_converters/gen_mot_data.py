import argparse
import glob
import os
import os.path as osp

import numpy as np


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


def parse_args():
    parser = argparse.ArgumentParser(description="Split Mot to train and test")
    parser.add_argument(
        "-s",
        "--src-data-path",
        type=str,
        required=True,
        help="src data path",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        required=True,
        help="output path",
    )
    return parser.parse_args()


def split_half_data(seq_root, out_dir):

    seqs = [s for s in sorted(os.listdir(seq_root)) if s.endswith("SDP")]

    # split_names = ["train", "test"]
    tid_train_curr = 0
    tid_train_last = -1
    tid_test_curr = 0
    tid_test_last = -1

    for seq in seqs:
        seq_info = open(osp.join(seq_root, seq, "seqinfo.ini")).read()
        seq_width = int(
            seq_info[
                seq_info.find("imWidth=") + 8 : seq_info.find("\nimHeight")
            ]
        )
        seq_height = int(
            seq_info[seq_info.find("imHeight=") + 9 : seq_info.find("\nimExt")]
        )

        len_frame = int(
            seq_info[
                seq_info.find("seqLength=") + 10 : seq_info.find("\nimWidth")
            ]
        )
        seq_path = os.path.join(seq_root, seq, "img1")
        images = sorted(glob.glob(seq_path + "/*.jpg"))

        half_frames = int(len_frame / 2)

        train_images = images[:half_frames]
        test_images = images[half_frames:]

        train_img_dir = osp.join(out_dir, "train", seq, "img1")
        test_img_dir = osp.join(out_dir, "test", seq, "img1")
        mkdirs(train_img_dir)
        mkdirs(test_img_dir)

        for src in train_images:
            os.system(f"cp -rf {src} {train_img_dir}")

        for src in test_images:
            os.system(f"cp -rf {src} {test_img_dir}")

        gt_txt = osp.join(seq_root, seq, "gt", "gt.txt")

        train_gt_dir = osp.join(out_dir, "train", seq, "gt")
        test_gt_dir = osp.join(out_dir, "test", seq, "gt")
        mkdirs(train_gt_dir)
        mkdirs(test_gt_dir)

        new_gt_train_txt = osp.join(train_gt_dir, "gt.txt")
        new_gt_test_txt = osp.join(test_gt_dir, "gt.txt")

        new_gt_train_with_ids_dir = os.path.join(
            os.path.dirname(train_gt_dir), "labels_with_ids"
        )
        new_gt_test_with_ids_dir = os.path.join(
            os.path.dirname(test_gt_dir), "labels_with_ids"
        )

        mkdirs(new_gt_train_with_ids_dir)
        mkdirs(new_gt_test_with_ids_dir)

        new_gt_train = open(new_gt_train_txt, "a+")
        new_gt_test = open(new_gt_test_txt, "a+")

        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=",")

        for fid, tid, x, y, w, h, mark, label, sss in gt:
            # for txt in all_txts:
            frame = int(fid)
            # bb = txt.split(',')
            if frame <= half_frames:
                line = "{},{},{},{},{},{},{},{},{}\n".format(
                    frame, tid, x, y, w, h, mark, label, sss
                )
                new_gt_train.write(line)

                if mark == 0 or not label == 1:
                    continue
                fid = int(fid)
                tid = int(tid)
                if not tid == tid_train_last:
                    tid_train_curr += 1
                    tid_train_last = tid
                x += w / 2
                y += h / 2
                label_fpath = osp.join(
                    new_gt_train_with_ids_dir, "{:06d}.txt".format(fid)
                )
                label_str = "0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                    tid_train_curr,
                    x / seq_width,
                    y / seq_height,
                    w / seq_width,
                    h / seq_height,
                )
                with open(label_fpath, "a") as f:
                    f.write(label_str)
            else:
                frame -= half_frames
                line = "{},{},{},{},{},{},{},{},{}\n".format(
                    frame, tid, x, y, w, h, mark, label, sss
                )
                new_gt_test.write(line)

                if mark == 0 or not label == 1:
                    continue
                fid = int(fid)
                tid = int(tid)
                if not tid == tid_test_last:
                    tid_test_curr += 1
                    tid_test_last = tid
                x += w / 2
                y += h / 2
                label_fpath = osp.join(
                    new_gt_test_with_ids_dir, "{:06d}.txt".format(fid)
                )
                label_str = "0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                    tid_test_curr,
                    x / seq_width,
                    y / seq_height,
                    w / seq_width,
                    h / seq_height,
                )
                with open(label_fpath, "a") as f:
                    f.write(label_str)

        new_gt_train.close()
        new_gt_test.close()
        # old_file.close()


if __name__ == "__main__":
    args = parse_args()
    split_half_data(args.src_data_path, args.out_dir)

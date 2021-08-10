# -*- coding:utf-8 -*-
#from F2M_model_V14 import *
from F2M_model_V14_2 import *
from random import shuffle, random
from collections import Counter
from imutils import face_utils

import tensorflow_io as tfio
import matplotlib.pyplot as plt
import numpy as np
import easydict
import os
import cv2
import dlib

FLAGS = easydict.EasyDict({"img_size": 256, 
                           
                           "load_size": 276,

                           "tar_size": 256,

                           "tar_load_size": 276,
                           
                           "batch_size": 1,
                           
                           "epochs": 200,
                           
                           "lr": 0.0002,
                           
                           "A_txt_path": "D:/[1]DB/[1]second_paper_DB/MegaAge_16_69_fullDB/trainA.txt",
                           
                           "A_img_path": "D:/[1]DB/[1]second_paper_DB/original_MegaAge_asian/megaage_asian/megaage_asian/train/",
                           
                           "B_txt_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-F_Morph-M_40_63_16_39/train/male_16_39_train.txt",
                           
                           #"B_img_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/Morph/All/male_16_39/",
                           "B_img_path": "D:/[1]DB/[1]second_paper_DB/original_MORPH/MORPH/EntireMORPH/Album2/",

                           "age_range": [40, 64],

                           "n_classes": 256,

                           "train": True,
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "",
                           
                           "save_checkpoint": "",
                           
                           "sample_images": "C:/Users/Yuhwan/Pictures/img2",

                           "shape_predict": "C:/Users/Yuhwan/Documents/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat",
                           
                           "A_test_txt_path": "",
                           
                           "A_test_img_path": "",
                           
                           "B_test_txt_path": "",
                           
                           "B_test_img_path": "",
                           
                           "test_dir": "A2B",
                           
                           "fake_B_path": "",
                           
                           "fake_A_path": ""})

g_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
d_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)



def input_func(A_data, B_data):

    A_img = tf.io.read_file(A_data[0])
    A_img = tf.image.decode_jpeg(A_img, 3)
    A_img = tf.image.resize(A_img, [FLAGS.img_size, FLAGS.img_size])
    #A_img = tf.image.random_crop(A_img, [FLAGS.img_size, FLAGS.img_size, 3])
    #A_img = A_img / 127.5 - 1.

    B_img = tf.io.read_file(B_data[0])
    B_img = tf.image.decode_jpeg(B_img, 3)
    B_img = tf.image.resize(B_img, [FLAGS.tar_size, FLAGS.tar_size])
    #B_img = tf.image.random_crop(B_img, [FLAGS.tar_size, FLAGS.tar_size, 3])
    #B_img = B_img / 127.5 - 1.

    if random() > 0.5:
        A_img = tf.image.flip_left_right(A_img)
        B_img = tf.image.flip_left_right(B_img)

    B_lab = int(B_data[1])
    A_lab = int(A_data[1])

    return A_img, A_lab, B_img, B_lab

def te_input_func(img, lab):

    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])

    lab = lab

    return img, lab

#@tf.function
def model_out(model, images, training=True):
    return model(images, training=training)

def decreas_func(x):
    return tf.maximum(0, tf.math.exp(x * (-2.77 / 100)))

def increase_func(x):
    x = tf.cast(tf.maximum(1, x), tf.float32)
    return tf.math.log(x + 1e-7)

def cal_loss(A2B_G_model, B2A_G_model, A_discriminator, B_discriminator,
             A_batch_images, B_batch_images, B_batch_labels, A_batch_labels):

    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_B, DB_fake  = model_out(A2B_G_model, A_batch_images, True)
        fake_A_ = model_out(B2A_G_model, tf.nn.tanh(fake_B), True)

        fake_A, DA_fake = model_out(B2A_G_model, B_batch_images, True)
        fake_B_ = model_out(A2B_G_model, tf.nn.tanh(fake_A), True)
        # DA_real, DB_real ?? ??? ?????ؾ??ұ??
        # ???? ???¿????? real discrim?? ?????ϱⰡ ?ָ??ҵ??ϴ?

        # 교수님이 말씀하신것 하고난 후 이어서 짤것 content loss도 수정하고 Cycleloss도 내가 github 에 써놓은대로 할것!


        # 원본의 landmark를 뽑은 후, 그 랜드마크와 좌표를 생성된 영상에도 똑같이 적용하고
        # 원본의 facial을 생성된곳에 warp해줌. 괜찮은 방법이 아니겠는가?





        DB_real = model_out(B_discriminator, B_batch_images, True)
        DB_fake = model_out(B_discriminator, tf.nn.tanh(fake_B[:, :, :, 0:3]), True)
        DA_real = model_out(A_discriminator, A_batch_images, True)
        DA_fake = model_out(A_discriminator, tf.nn.tanh(fake_A[:, :, :, 0:3]), True)

        ################################################################################################
        # ???̿? ???? distance?? ???ϴ°?
        return_loss = 0.
        for i in range(FLAGS.batch_size):   # ?????? ?????ͷ??Ϸ??? compare label?? ?ϳ? ?? ???????? ?Ѵ? ??????!!!!
            energy_ft = tf.reduce_sum(tf.abs(tf.reduce_mean(fake_A[i, :, :, 3:], [0,1]) - tf.reduce_mean(fake_B[:, :, :, 3:], [1,2])), 1)
            energy_ft2 = tf.reduce_sum(tf.abs(tf.reduce_mean(fake_A_[i, :, :, 3:], [0,1]) - tf.reduce_mean(fake_B_[:, :, :, 3:], [1,2])), 1)
            compare_label = tf.subtract(A_batch_labels, B_batch_labels[i])

            T = 4
            label_buff = tf.less(tf.abs(compare_label), T)
            label_cast = tf.cast(label_buff, tf.float32)

            realB_fakeB_loss = label_cast * increase_func(energy_ft) \
                + (1 - label_cast) * 5 * decreas_func(energy_ft)

            realA_fakeA_loss = label_cast * increase_func(energy_ft2) \
                + (1 - label_cast) * 5 * decreas_func(energy_ft2)

            # A?? B ???̰? ?ٸ??? ?????Լ?, ?????? ?????Լ?

            loss_buf = 0.
            for j in range(FLAGS.batch_size):
                loss_buf += realB_fakeB_loss[j] + realA_fakeA_loss[j]
            loss_buf /= FLAGS.batch_size

            return_loss += loss_buf
        return_loss /= FLAGS.batch_size
        ################################################################################################
        # content loss ?? ?ۼ?????
        f_B = tf.nn.tanh(fake_B[:, :, :, 0:3])
        f_B_x, f_B_y = tf.image.image_gradients(f_B)
        f_B_m = tf.add(tf.abs(f_B_x), tf.abs(f_B_y))
        f_B = tf.abs(f_B - f_B_m)

        f_A = tf.nn.tanh(fake_A[:, :, :, 0:3])
        f_A_x, f_A_y = tf.image.image_gradients(f_A)
        f_A_m = tf.add(tf.abs(f_A_x), tf.abs(f_A_y))
        f_A = tf.abs(f_A - f_A_m)

        r_A = A_batch_images
        r_A_x, r_A_y = tf.image.image_gradients(r_A)
        r_A_m = tf.add(tf.abs(r_A_x), tf.abs(r_A_y))
        r_A = tf.abs(r_A - r_A_m)

        r_B = B_batch_images
        r_B_x, r_B_y = tf.image.image_gradients(r_B)
        r_B_m = tf.add(tf.abs(r_B_x), tf.abs(r_B_y))
        r_B = tf.abs(r_B - r_B_m)

        id_loss = tf.reduce_mean(tf.abs(f_B - r_A)) * 5.0 \
            + tf.reduce_mean(tf.abs(f_A - r_B)) * 5.0   # content loss

        Cycle_loss = (tf.reduce_mean(tf.abs(tf.nn.tanh(fake_A_[:, :, :, 0:3]) - A_batch_images))) \
            * 10.0 + (tf.reduce_mean(tf.abs(tf.nn.tanh(fake_B_[:, :, :, 0:3]) - B_batch_images))) * 10.0
        G_gan_loss = tf.reduce_mean((DB_fake - tf.ones_like(DB_fake))**2) \
            + tf.reduce_mean((DA_fake - tf.ones_like(DA_fake))**2)

        Adver_loss = (tf.reduce_mean((DB_real - tf.ones_like(DB_real))**2) + tf.reduce_mean((DB_fake - tf.zeros_like(DB_fake))**2)) / 2. \
            + (tf.reduce_mean((DA_real - tf.ones_like(DA_real))**2) + tf.reduce_mean((DA_fake - tf.zeros_like(DA_fake))**2)) / 2.

        g_loss = Cycle_loss + G_gan_loss + return_loss + id_loss
        d_loss = Adver_loss

    g_grads = g_tape.gradient(g_loss, A2B_G_model.trainable_variables + B2A_G_model.trainable_variables)
    d_grads = d_tape.gradient(d_loss, A_discriminator.trainable_variables + B_discriminator.trainable_variables)

    g_optim.apply_gradients(zip(g_grads, A2B_G_model.trainable_variables + B2A_G_model.trainable_variables))
    d_optim.apply_gradients(zip(d_grads, A_discriminator.trainable_variables + B_discriminator.trainable_variables))

    return g_loss, d_loss

def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

def main():

    pre_trained_encoder1 = tf.keras.applications.ResNet50V2(include_top=False, input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    pre_trained_encoder2 = tf.keras.applications.VGG16(include_top=False, input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    pre_trained_encoder2.summary()

    A2B_G_model = F2M_generator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    B2A_G_model = F2M_generator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))

    face_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(FLAGS.shape_predict)

    A2B_G_model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(A2B_G_model=A2B_G_model, B2A_G_model=B2A_G_model,
                                   g_optim=g_optim, d_optim=d_optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored!!")
    else:
        A2B_G_model.get_layer("conv_en_1").set_weights(pre_trained_encoder1.get_layer("conv1_conv").get_weights())
        B2A_G_model.get_layer("conv_en_1").set_weights(pre_trained_encoder1.get_layer("conv1_conv").get_weights())
    
        A2B_G_model.get_layer("conv_en_3").set_weights(pre_trained_encoder2.get_layer("block2_conv1").get_weights())
        B2A_G_model.get_layer("conv_en_3").set_weights(pre_trained_encoder2.get_layer("block2_conv1").get_weights())

        A2B_G_model.get_layer("conv_en_5").set_weights(pre_trained_encoder2.get_layer("block3_conv1").get_weights())
        B2A_G_model.get_layer("conv_en_5").set_weights(pre_trained_encoder2.get_layer("block3_conv1").get_weights())


    if FLAGS.train:
        count = 0

        A_images = np.loadtxt(FLAGS.A_txt_path, dtype="<U100", skiprows=0, usecols=0)
        A_images = [FLAGS.A_img_path + data for data in A_images]
        A_labels = np.loadtxt(FLAGS.A_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        B_images = np.loadtxt(FLAGS.B_txt_path, dtype="<U100", skiprows=0, usecols=0)
        B_images = [FLAGS.B_img_path + data for data in B_images]
        B_labels = np.loadtxt(FLAGS.B_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        alpha = 0.5

        for epoch in range(FLAGS.epochs):
            min_ = min(len(A_images), len(B_images))
            A = list(zip(A_images, A_labels))
            B = list(zip(B_images, B_labels))
            shuffle(B)
            shuffle(A)
            B_images, B_labels = zip(*B)
            A_images, A_labels = zip(*A)
            A_images = A_images[:min_]
            A_labels = A_labels[:min_]
            B_images = B_images[:min_]
            B_labels = B_labels[:min_]

            A_zip = np.array(list(zip(A_images, A_labels)))
            B_zip = np.array(list(zip(B_images, B_labels)))

            # ?????? ???̿? ???ؼ? distance?? ???ϴ? loss?? ?????ϸ?, ?ᱹ???? ?ش??̹????? ???̸? ?״??? ?????ϴ? ȿ????? ??????????
            gener = tf.data.Dataset.from_tensor_slices((A_zip, B_zip))
            gener = gener.shuffle(len(B_images))
            gener = gener.map(input_func)
            gener = gener.batch(FLAGS.batch_size)
            gener = gener.prefetch(tf.data.experimental.AUTOTUNE)

            train_idx = min_ // FLAGS.batch_size
            train_it = iter(gener)
            
            for step in range(train_idx):
                A_batch_images, A_batch_labels, B_batch_images, B_batch_labels = next(train_it)
                A_batch_bgr = tfio.experimental.color.rgb_to_bgr(tf.image.resize(A_batch_images, [512, 512]))
                B_batch_bgr = tfio.experimental.color.rgb_to_bgr(tf.image.resize(B_batch_images, [512, 512]))
                for i in range(FLAGS.batch_size):
                    A_img = A_batch_bgr[i]
                    A_img = tf.cast(A_img, tf.uint8).numpy()
                    B_img = B_batch_bgr[i]
                    B_img = tf.cast(B_img, tf.uint8).numpy()
                    A_gray = cv2.cvtColor(A_img, cv2.COLOR_BGR2GRAY)
                    A_gray = np.array(A_gray, np.uint8)
                    B_gray = cv2.cvtColor(B_img, cv2.COLOR_BGR2GRAY)
                    B_gray = np.array(B_gray, np.uint8)

                    A_rects = face_detector(A_gray, 1)
                    B_rects = face_detector(B_gray, 1)

                    for (j, rect) in enumerate(A_rects):
                        shape = predictor(A_gray, rect)
                        shape = face_utils.shape_to_np(shape)

                        (x, y, w, h) = face_utils.rect_to_bb(rect)
                        A_src = []
                        for (x,y) in shape:
                            A_src.append((int(x), int(y)))

                        subdiv = cv2.Subdiv2D((0, 0, 512, 512))
                        for p in A_src:
                            print(p)
                            subdiv.insert(p)    # 된다!

                        triangleList = subdiv.getTriangleList();
                        size = FLAGS.img_size
                        for t in triangleList:

                            pt1 = (t[0], t[1])
                            pt2 = (t[2], t[3])
                            pt3 = (t[4], t[5])
                            
                            if rect_contains((0, 0, 512, 512), pt1) and rect_contains((0, 0, 512, 512), pt2) and rect_contains((0, 0, 512, 512), pt3):
                                cv2.line(A_img, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA, 0)
                                cv2.line(A_img, pt2, pt3, (0, 0, 255), 1, cv2.LINE_AA, 0)
                                cv2.line(A_img, pt3, pt1, (0, 0, 255), 1, cv2.LINE_AA, 0)

                        cv2.imshow("ttt", A_img)
                        cv2.waitKey(0)

                    for (j, rect) in enumerate(B_rects):
                        shape = predictor(B_gray, rect)
                        shape = face_utils.shape_to_np(shape)

                        (x, y, w, h) = face_utils.rect_to_bb(rect)
                        B_src = []
                        for (x,y) in shape:
                            B_src.append((int(x), int(y)))

                        subdiv = cv2.Subdiv2D((0, 0, 512, 512))
                        for p in B_src:
                            print(p)
                            subdiv.insert(p)    # 된다!

                        triangleList = subdiv.getTriangleList();
                        size = FLAGS.img_size
                        for t in triangleList:

                            pt1 = (t[0], t[1])
                            pt2 = (t[2], t[3])
                            pt3 = (t[4], t[5])
                            
                            if rect_contains((0, 0, 512, 512), pt1) and rect_contains((0, 0, 512, 512), pt2) and rect_contains((0, 0, 512, 512), pt3):
                                cv2.line(B_img, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA, 0)
                                cv2.line(B_img, pt2, pt3, (0, 0, 255), 1, cv2.LINE_AA, 0)
                                cv2.line(B_img, pt3, pt1, (0, 0, 255), 1, cv2.LINE_AA, 0)

                        cv2.imshow("tt", B_img)
                        cv2.waitKey(0)

                    # triangleList 를 어느것을 기준으로하느냐가 중요함
                    # img1, img2, imgMorph, t1, t2, t, alpha
                    imgMorph = np.zeros([512, 512, 3], dtype=np.uint8)
                    points = []
                    for j in range(len(shape)):
                        x = (1 - alpha) * A_src[j][0] + alpha * B_src[j][0]
                        y = (1 - alpha) * A_src[j][1] + alpha * B_src[j][1]
                        points.append((x, y))

                    for List in triangleList:   # B 기준
                        pt1 = (int(List[0]), int(List[1]))
                        pt2 = (int(List[2]), int(List[3]))
                        pt3 = (int(List[4]), int(List[5]))

                        t1 = [A_src[pt1], A_src[pt2], A_src[pt3]]   # 여기는 내일 이어서 다시해보자!
                        t2 = [B_src[pt1], B_src[pt2], B_src[pt3]]
                        t = [points[pt1], points[pt2], points[pt3]]

                        r1 = cv2.boundingRect(np.float32([t1]))
                        r2 = cv2.boundingRect(np.float32([t2]))
                        r = cv2.boundingRect(np.float32([t]))

                        # Offset points by left top corner of the respective rectangles
                        t1Rect = []
                        t2Rect = []
                        tRect = []

                        for i in range(0, 3):
                            tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
                            t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
                            t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

                        # Get mask by filling triangle
                        mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
                        cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

                        # Apply warpImage to small rectangular patches
                        img1Rect = A_img[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
                        img2Rect = B_img[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

                        size = (r[2], r[3])
                        warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
                        warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

                        # Alpha blend rectangular patches
                        imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

                        # Copy triangular region of the rectangular patch to the output image
                        img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask

                    
                    # https://github.com/spmallick/learnopencv/blob/master/FaceMorph/faceMorph.py
                    # https://learnopencv.com/face-morph-using-opencv-cpp-python/#id1444128263
                    # https://learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/


                g_loss, d_loss = cal_loss(A2B_G_model, B2A_G_model, A_discriminator, B_discriminator,
                                          A_batch_images, B_batch_images, B_batch_labels, A_batch_labels)

                print("Epoch = {}[{}/{}];\nStep(iteration) = {}\nG_Loss = {}, D_loss = {}".format(epoch,step,train_idx,
                                                                                                  count+1,
                                                                                                  g_loss, d_loss))
                
                if count % 100 == 0:
                    fake_B = model_out(A2B_G_model, A_batch_images, False)
                    fake_A = model_out(B2A_G_model, B_batch_images, False)

                    plt.imsave(FLAGS.sample_images + "/fake_B_{}.jpg".format(count), tf.nn.tanh(fake_B[0, :, :, 0:3]) * 0.5 + 0.5)
                    plt.imsave(FLAGS.sample_images + "/fake_A_{}.jpg".format(count), tf.nn.tanh(fake_A[0, :, :, 0:3]) * 0.5 + 0.5)
                    plt.imsave(FLAGS.sample_images + "/real_B_{}.jpg".format(count), B_batch_images[0] * 0.5 + 0.5)
                    plt.imsave(FLAGS.sample_images + "/real_A_{}.jpg".format(count), A_batch_images[0] * 0.5 + 0.5)


                #if count % 1000 == 0:
                #    num_ = int(count // 1000)
                #    model_dir = "%s/%s" % (FLAGS.save_checkpoint, num_)
                #    if not os.path.isdir(model_dir):
                #        print("Make {} folder to store the weight!".format(num_))
                #        os.makedirs(model_dir)
                #    ckpt = tf.train.Checkpoint(A2B_G_model=A2B_G_model, B2A_G_model=B2A_G_model,
                #                               A_discriminator=A_discriminator, B_discriminator=B_discriminator,
                #                               g_optim=g_optim, d_optim=d_optim)
                #    ckpt_dir = model_dir + "/F2M_V8_{}.ckpt".format(count)
                #    ckpt.save(ckpt_dir)

                count += 1

    else:
        if FLAGS.test_dir == "A2B": # train data?? A?? ?ƴ? B?? ?ؾ???
            A_train_data = np.loadtxt(FLAGS.A_txt_path, dtype="<U100", skiprows=0, usecols=0)
            A_train_data = [FLAGS.A_img_path + data for data in A_train_data]
            A_train_label = np.loadtxt(FLAGS.A_txt_path, dtype=np.int32, skiprows=0, usecols=1)

            A_test_data = np.loadtxt(FLAGS.A_test_txt_path, dtype="<U200", skiprows=0, usecols=0)
            A_test_data = [FLAGS.A_test_img_path + data for data in A_test_data]
            A_test_label = np.loadtxt(FLAGS.A_test_txt_path, dtype=np.int32, skiprows=0, usecols=1)

            tr_gener = tf.data.Dataset.from_tensor_slices((A_train_data, A_train_label))
            tr_gener = tr_gener.map(te_input_func)
            tr_gener = tr_gener.batch(1)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            te_gener = tf.data.Dataset.from_tensor_slices((A_test_data, A_test_label))
            te_gener = te_gener.map(te_input_func)
            te_gener = te_gener.batch(1)
            te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_it = iter(tr_gener)
            tr_idx = len(A_train_data) // 1
            te_it = iter(te_gener)
            te_idx = len(A_test_data) // 1

            for i in range(te_idx):
                te_A_images, te_A_labels = next(te_it)
                fake_B, te_feature = model_out(A2B_G_model, te_A_images, False)    # [1, 256]
                te_features = te_feature[0]
                dis = []
                lab = []
                for j in range(tr_idx):
                    tr_A_images, tr_A_labels = next(tr_it)
                    _, tr_feature = model_out(A2B_G_model, tr_A_images, False)    # [1, 256]
                    tr_features = tr_feature[0]

                    d = tf.reduce_sum(tf.abs(tr_features - te_features), -1)
                    dis.append(d.numpy())
                    lab.append(tr_A_labels[0].numpy())

                min_distance = np.argmin(dis, axis=-1)
                generated_age = lab[min_distance]

                name = (A_test_data[i].split("/")[-1]).split(".")[0]
                plt.imsave(FLAGS.fake_B_path + "/" + name + "_{}".format(generated_age) + ".jpg", fake_B[0].numpy() * 0.5 + 0.5)



        if FLAGS.test_dir == "B2A": # train data ?? B?? ?ƴ? A?? ?ؾ???
            B_train_data = np.loadtxt(FLAGS.B_txt_path, dtype="<U100", skiprows=0, usecols=0)
            B_train_data = [FLAGS.B_img_path + data for data in B_train_data]
            B_train_label = np.loadtxt(FLAGS.B_txt_path, dtype=np.int32, skiprows=0, usecols=1)

            B_test_data = np.loadtxt(FLAGS.B_test_txt_path, dtype="<U200", skiprows=0, usecols=0)
            B_test_data = [FLAGS.B_test_img_path + data for data in B_test_data]
            B_test_label = np.loadtxt(FLAGS.B_test_txt_path, dtype="<U200", skiprows=0, usecols=1)

            tr_gener = tf.data.Dataset.from_tensor_slices((B_train_data, B_train_label))
            tr_gener = tr_gener.map(te_input_func)
            tr_gener = tr_gener.batch(1)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            te_gener = tf.data.Dataset.from_tensor_slices((B_test_data, B_test_label))
            te_gener = te_gener.map(te_input_func)
            te_gener = te_gener.batch(1)
            te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_it = iter(tr_gener)
            tr_idx = len(B_train_data) // 1
            te_it = iter(te_gener)
            te_idx = len(B_test_data) // 1

            for i in range(te_idx):
                te_B_images, te_B_labels = next(te_it)
                fake_A, te_feature = model_out(B2A_G_model, te_B_images, False)    # [1, 256]
                te_features = te_feature[0]
                dis = []
                lab = []
                for j in range(tr_idx):
                    tr_B_images, tr_B_labels = next(tr_it)
                    _, tr_feature = model_out(B2A_G_model, tr_B_images, False)    # [1, 256]
                    tr_features = tr_feature[0]

                    d = tf.reduce_sum(tf.abs(tr_features - te_features), -1)
                    dis.append(d.numpy())
                    lab.append(tr_B_labels[0].numpy())

                min_distance = np.argmin(dis, axis=-1)
                generated_age = lab[min_distance]

                name = (B_test_data[i].split("/")[-1]).split(".")[0]
                plt.imsave(FLAGS.fake_A_path + "/" + name + "_{}".format(generated_age) + ".jpg", fake_A[0].numpy() * 0.5 + 0.5)



if __name__ == "__main__":
    main()

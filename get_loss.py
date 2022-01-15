import keras.backend as K
import numpy as np
import tensorflow as tf

def get_ytrue_ypre(ypre,batch_size):
    mesh_x = tf.cast(tf.reshape(tf.tile(tf.range(ypre.shape[1]), [ypre.shape[2]]), (1, ypre.shape[1], ypre.shape[2], 1, 1)),tf.float32)
    mesh_y = tf.transpose(mesh_x, (0,2,1,3,4))
    mesh_xy = tf.tile(tf.concat([mesh_x,mesh_y],-1), [batch_size, 1, 1,ypre.shape[3], 1])

    # anchor_gird = K.constant(anchors,dtype='float32',shape=[1,1,1,3,2])
    # true_wh = K.exp(ytrue[...,2:4])*anchor_gird

    pre_xy = K.sigmoid(ypre[...,:2])+mesh_xy
    # pre_wh = K.exp(ypre[...,2:4])*anchor_gird
    pre_con = K.sigmoid(ypre[...,4])
    pre_cla = ypre[...,5:]

    ypre = tf.concat([pre_xy,ypre[...,2:4],K.expand_dims(pre_con,axis=-1),pre_cla],axis=-1)

    return ypre

def get_IOU(ytrue,ypre,input_size,ignore_thresh,anchors):
    rate = input_size/int(ytrue.shape[1])
    anchor_gird = K.constant(anchors, dtype='float32', shape=[1, 1, 1, 3, 2])

    true_xy = ytrue[..., :2] * rate
    true_wh = K.exp(ytrue[..., 2:4]) * anchor_gird
    true_wh_half = true_wh/ 2
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    pre_xy = ypre[..., :2] * rate
    pre_wh = K.exp(ypre[..., 2:4]) * anchor_gird
    pre_wh_half = pre_wh/2
    pre_mins =  pre_xy - pre_wh_half
    pre_maxes =  pre_xy + pre_wh_half

    intersect_mins  = K.maximum(pre_mins,  true_mins)
    intersect_maxes = K.minimum(pre_maxes, true_maxes)

    intersect_wh    = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pre_wh[..., 0] * pre_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    IOUS  = intersect_areas/union_areas

    return tf.cast(IOUS<ignore_thresh,tf.float32)

def get_loss_box(ytrue,ypre,box_scale,object_mask,batch_size):
    # xy_delta = box_scale*object_mask*K.binary_crossentropy(ytrue[...,:2],ypre[...,:2], from_logits=True)
    xy_delta = box_scale*object_mask*K.square(ytrue[...,0:2]-ypre[...,0:2])
    wh_delta = 0.5*box_scale*object_mask* K.square(ytrue[...,2:4]-ypre[...,2:4])
    loss_xy = K.sum(xy_delta)
    loss_wh = K.sum(wh_delta)

    return loss_xy,loss_wh

def get_loss_con(ytrue,ypre,noobj_scale,object_mask,ignore_mask,batch_size):
    object_mask = K.squeeze(object_mask,axis=-1)
    con_delta = object_mask*K.square(ypre-ytrue) + noobj_scale*(1-object_mask)*ypre*ignore_mask
    loss_con = K.sum(con_delta)

    return loss_con

def get_loss_c(ytrue,ypre,object_mask,batch_size):
    # class_delta = object_mask * (ypre-ytrue)
    # loss_class = K.sum(K.square(class_delta),list(range(1,5)))
    ytrue = tf.cast(ytrue, tf.int64)
    class_delta = object_mask * tf.expand_dims(tf.nn.softmax_cross_entropy_with_logits
                                              (labels=ytrue, logits=ypre), 4)
    loss_class = K.sum(class_delta)

    return loss_class

def lossCalculator(ytrue,ypre,anchors,batch_size,input_size,box_scale,noobj_scale,ignore_thresh):
    #调整ypre与ytrue的形状为[b,13,13,3,n]
    ypre  = K.reshape(ypre ,shape = [-1,ypre.shape[-3],ypre.shape[-2],anchors.shape[0],ypre.shape[-1]//anchors.shape[0]])
    ytrue = K.reshape(ytrue,shape = [-1,ypre.shape[1] ,ypre.shape[2] ,ypre.shape[3],ypre.shape[4]])

    ypre = get_ytrue_ypre(ypre,batch_size)
    #存在目标的mask
    object_mask = K.expand_dims(ytrue[...,4],4)
    #负样本mask
    ignore_mask = get_IOU(ytrue[...,:4],ypre[...,:4],input_size,ignore_thresh,anchors)

    loss_xy,loss_wh = get_loss_box(ytrue[...,:4],ypre[...,:4],box_scale,object_mask,batch_size)
    loss_con = get_loss_con(ytrue[...,4],ypre[...,4],noobj_scale,object_mask,ignore_mask,batch_size)
    loss_class = get_loss_c(ytrue[...,5:],ypre[...,5:],object_mask,batch_size)

    losses = loss_xy+loss_wh+loss_con+loss_class
    # print(losses)

    return losses

    # return losses,loss_xy,loss_wh,loss_con,loss_class,object_mask

def fn_loss(ytrues,ypres):
    ignore_thresh = 0.5
    noobj_scale = 0.5
    box_scale = 1
    input_size = 416
    batch_size = 1
    anchors = np.array([[[10,13], [16,30], [33,23]],
                        [[30,61], [62,45], [59,119]],
                        [[116,90], [156,198], [373,326]]])

    losses = [lossCalculator(ytrues[i], ypres[i], anchors[i], batch_size, input_size, box_scale, noobj_scale, ignore_thresh) for i in range(3)]

    # loss,loss_xy,loss_wh,loss_con,loss_class,object_mask= \
    #     lossCalculator(ytrues, ypres, anchors[2 - ypres.shape[1] // 26], batch_size,input_size, box_scale, noobj_scale,ignore_thresh)

    # loss = tf.Print(loss, [loss,loss_xy,loss_wh, loss_con, loss_class, K.sum(object_mask)],
    #                 message='loss: ')
    loss = tf.sqrt(K.sum(losses))
    return loss


from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.layers import Conv2D,ZeroPadding2D,Add,UpSampling2D,Concatenate

def conv(*args,**kwargs):
    new_kwargs = {'kernel_regularizer':l2(5e-4)}
    new_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2,2) else 'same'
    new_kwargs.update(kwargs)

    return Conv2D(*args,**new_kwargs)

def CBL(x,*args,**kwargs):
    new_kwargs ={ 'use_bias': False}
    new_kwargs.update(kwargs)
    x = conv(*args,**new_kwargs)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    return x

def PCBL(x,num_filters):
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = CBL(x,num_filters,(3,3),strides=(2,2))

    return x

def CBLR(x,num_filters):
    y = CBL(x,num_filters,(1,1))
    y = CBL(y,num_filters*2,(3,3))
    x = Add()([x,y])

    return x

def CBL5(x,num_filters):
    x = CBL(x, num_filters, (1,1))
    x = CBL(x, num_filters*2, (3,3))
    x = CBL(x, num_filters, (1,1))
    x = CBL(x, num_filters*2, (3,3))
    x = CBL(x, num_filters, (1,1))

    return x

def CBLC(x,num_filters,out_filters):
    x = CBL(x, num_filters * 2, (3, 3))
    x = conv(out_filters,(1,1))(x)

    return x

def CBLU(x, num_filters):
    x = CBL(x, num_filters, (1, 1))
    x = UpSampling2D(2)(x)

    return x

def body(inputs,num_anchors,num_classes):
    out = []
    x = CBL(inputs,32,(3,3))
    n = [1,2,8,8,4]
    for i in range(5):
        x = PCBL(x,2**(6+i))
        for _ in range(n[i]):
            x = CBLR(x,2**(5+i))

        if i in [2,3,4]:
            out.append(x)

    x1 = CBL5(out[2],512)
    y1 = CBLC(x1,512,num_anchors*(num_classes+5))

    x = CBLU(x1,256)
    x = Concatenate()([x,out[1]])

    x2 = CBL5(x,256)
    y2 = CBLC(x2, 256, num_anchors * (num_classes + 5))

    x = CBLU(x2,128)
    x = Concatenate()([x, out[0]])

    x3 = CBL5(x,128)
    y3 = CBLC(x3, 128, num_anchors * (num_classes + 5))

    return [y1,y2,y3]


from keras.layers import Input
from keras.models import Model

def main():
    model_input = Input(shape=(416, 416, 3))
    model_output = body(model_input, 3, 80)
    model = Model(model_input, model_output)
    model.summary()



if __name__ == '__main__':
    main()















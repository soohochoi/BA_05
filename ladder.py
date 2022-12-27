#모델을 만들기위해 keras를 불러옴
import keras
from keras.models import *
from keras.layers import *

import tensorflow as tf

#각 레이어마다 베타값을 생성해주는 함수
class init_Beta(Layer):
    def __init__(self  , **kwargs):
        super(init_Beta, self).__init__(**kwargs)
        
    def build(self, input_shape):
        if self.built:
            return
        
        self.beta = self.add_weight(name='beta', 
                                      shape= input_shape[1:] ,
                                      initializer='zeros',
                                      trainable=True)
        self.built = True
        super(init_Beta, self).build(input_shape)
        
    def call(self, x, training=None):
        return tf.add(x, self.beta)

#인코딩에 대한 값을 디코딩해주기 위한 함수
class Guass(Layer):
    def __init__(self , **kwargs):
        super(Guass, self).__init__(**kwargs)
        
    def wi(self, init, name):
        if init == 1:
            return self.add_weight(name='guess_'+name, 
                                      shape=(self.size,),
                                      initializer='ones',
                                      trainable=True)
        elif init == 0:
            return self.add_weight(name='guess_'+name, 
                                      shape=(self.size,),
                                      initializer='zeros',
                                      trainable=True)
        else:
            raise ValueError("Guass의 레이어의 error임 '%d' " % init)

#레이어에 훈련 가능한 가중치 변수를 생성하는 함수
    def build(self, input_shape):

        self.size = input_shape[0][-1]

        init_values = [0., 1., 0., 0., 0., 0., 1., 0., 0., 0.]
        self.a = [self.wi(v, 'a' + str(i + 1)) for i, v in enumerate(init_values)]
        super(Guass , self).build(input_shape)

    #호출함
    def call(self, x):
        z_c, u = x 
        #sigmoid를 계산
        def compute(y):
            return y[0] * tf.sigmoid(y[1] * u + y[2]) + y[3] * u + y[4]

        mu = compute(self.a[:5])
        v = compute(self.a[5:])

        z_estimation = (z_c - mu) * v + mu
        return z_estimation
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.size)

# 배치 정규화 함수
def batch_normalization(batch, mean=None, var=None):
    if mean is None or var is None:
        mean, var = tf.nn.moments(batch, axes=[0])
    return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))

#노이즈를 추가하는 함수
def add_noise( inputs , noise_std ):
    return Lambda( lambda x: x + tf.random.normal(tf.shape(x)) * noise_std  )( inputs )

#위에 정의된 함수를 바탕으로 ladder 네트워크를 생성그리고 하이퍼파라미터 설정
def ladder_network(size_of_layers=[784, 1000, 500, 250, 250, 250, 10],
     noise_std=0.3,
     denoising_cost=[1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10]):
    #레이어의 갯수 생성
    L = len(size_of_layers) - 1

    inputs_l = Input((size_of_layers[0],))
    inputs_u = Input((size_of_layers[0],))

    fc_enc = [Dense(s, use_bias=False, kernel_initializer='glorot_normal') for s in size_of_layers[1:] ]
    fc_dec = [Dense(s, use_bias=False, kernel_initializer='glorot_normal') for s in size_of_layers[:-1]]
    betas  = [init_Beta() for l in range(L)]
    #인코더 정의하기
    def encoder(inputs, noise_std  ):
        h = add_noise(inputs, noise_std)
        all_z    = [None for _ in range( len(size_of_layers))]
        all_z[0] = h
        
        for l in range(1, L+1):
            z_pre = fc_enc[l-1](h)
            z = Lambda(batch_normalization)(z_pre)
            z = add_noise (z, noise_std)

            if l == L:
                h = Activation('softmax')(betas[l-1](z))
            else:
                h = Activation('relu')(betas[l-1](z))
                
            all_z[l] = z

        return h, all_z

    y_c_l, _ = encoder(inputs_l, noise_std)
    y_l, _ = encoder(inputs_l, 0.0)

    y_c_u, corr_z = encoder(inputs_u , noise_std)
    y_u,  clean_z = encoder(inputs_u , 0.0)

    # 디코더 정의 및  코스트 저장 리스트 생성
    d_cost = []
    for l in range(L, -1, -1):
        z, z_c = clean_z[l], corr_z[l]
        if l == L:
            u = y_c_u
        else:
            u = fc_dec[l]( z_estimation )
        u = Lambda(batch_normalization)(u)
        z_estimation  = Guass()([z_c, u])
        d_cost.append((tf.reduce_mean(tf.reduce_sum(tf.square(z_estimation - z), 1)) / size_of_layers[l]) * denoising_cost[l])

    u_cost = tf.add_n(d_cost)

    y_c_l = Lambda(lambda x: x[0])([y_c_l, y_l, y_c_u, y_u, u, z_estimation, z])

    train_model = Model([inputs_l, inputs_u], y_c_l)
    train_model.add_loss(u_cost)
    # adam 활용해 최적화 시키고
    train_model.compile(keras.optimizers.Adam(lr=0.02 ), 'categorical_crossentropy', metrics=['accuracy'])
    #dense loss추출하도록 함
    train_model.metrics_names.append("final_loss")
    train_model.metrics_tensors.append(u_cost)
    te_m = Model(inputs_l, y_l)
    train_model.test_model = te_m

    return train_model
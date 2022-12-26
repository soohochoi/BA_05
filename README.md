# BA_05 Semi-supervised Learning

<p align="center"><img width="604" alt="image" src="https://user-images.githubusercontent.com/97882448/208838979-8e8275d1-1cba-46e7-a584-da075e5a9c54.png">

이자료는 고려대학교 비니지스 애널리틱스 강필성교수님께 배운 Semi-supervised learning을 바탕으로 만들어졌습니다.
먼저, Semi-supervised learning의 방식의 기본적인 개념을 배운후에 Consistency regularization개념 및 코드를 통해 직접 구현을 해봄으로써 이해를 돕도록하겠습니다.

## BA_05 Semi-supervised learning 개념설명

<p align="center"><img width="600" alt="image" src="https://user-images.githubusercontent.com/97882448/209476685-f981bafd-fc68-40a5-9eb9-3460d7e6f185.png">

위 그림처럼 현실에서 데이터에 대한 답이 있는 경우를 예측할 때 supervised learning이라고 하고 답이 없는 경우는 unsupervised learning이라고 합니다. 그러나 우리가 살아가는 세계에는 데이터의 라벨이 조금밖에 없는경우도 다수로 존재합니다. 이럴때 라벨이 조금만 없으면 그것을 제외시켜도 되지만 다수인경우 데이터를 버리기에는 데이터가 아깝습니다. 또한 위에서 오른쪽 그림을 보시면 데이터를 사용해서 분류성능을 더욱 높일 수 있습니다. 단 모든경우에 그렇지는 않기 때문에 분류성능을 높이기 위해서 Pseudo-labeling methods와  Consistency regularization, Hybrid methods와 같은 방법론들이 나오고 있습니다.

## BA_05 Semi-supervised learning-Consistency regularization
<p align="center"><img width="900" alt="image" src="https://user-images.githubusercontent.com/97882448/209476776-5b7927cf-4929-4ded-823f-9f4c1e252f21.png">

Consistency regularization이란 Unlabeled data point에 작은 변화를 주어도 예측의 결과에는 일관성이 있을 것이라는 가정에서 출발합니다. 또한 Unlabeled data는 예측결과를 알 수 없기 때문에 data augmentation을 통해 class가 바뀌지 않을 정도의 변화를 줬을 때, 원래 데이터와의 예측결과가 같아지도록 unsupervised loss를 주어 학습합니다.

## Consistency regularization-Ladder network
  
<p align="center"><img width="900" alt="image" src="https://user-images.githubusercontent.com/97882448/209515640-c6b0c580-f9a8-4b7e-8dcd-1559ccb8e7d7.png">

Ladder network란 지도학습과 비지도학습을 결합한 딥러닝 모형입니다.
위쪽그림을 보시면 비지도 학습 pre-training에서 끝나지 않고 지도학습과 함께 training하며 2개의 인코더와 1개의 디코더로 이루어집니다.
Hierarchical latent variable model의 계층적인 특징을 반영한 오토 인코더를 사용하였고 층들의 연결을 Deterministic이 아니라 Stochastic으로 바꿔 주는것이 특징이며 효과적인 학습을 위해 Denoising기법을 활용한 모양이 꼭 사다리와 닮아 있어 Ladder network라고 합니다. Denoising이란 잡음을 추가한 데이터를 학습하여 데이터가 가지고 있는 본래의 고유한 특징을 더 잘 찾기 위한 방법입니다. 
  
구조를 조금더 자세히 보면 먼저 라벨이 있는 데이터는 가우시안 노이즈를 넣은 corrupted  f path와 Clean f path의 값이 빨간색처럼 유사하게 학습을합니다. 또한 corrupted f path가 Denoising q path로 넘어가서 하늘색처럼 Clean f path와  hidden state가 유사하도록 학습합니다. 라벨이 없는 데이터는 답이 없기 때문에 하늘색과 같은 loss만 최소화 시키는 것으로 학습이 진행되고 이러한 구조를 변형해서 디코더에서 가장 높은 layer만  사용하는 모델인 𝛾−모델이 있습니다

## Data set 소개
  ### Mnist
  
 <p align="center"><img width="900" alt="image" src="https://user-images.githubusercontent.com/97882448/209575433-8e15ddfb-2234-47b7-8276-64dee9e86ce8.png">

MNIST는 숫자 0부터 9까지의 이미지로 구성된 손글씨 데이터셋입니다. 이 데이터의 기원은 과거에 우체국에서 편지의 우편 번호를 인식하기 위해서 만들어진 훈련 데이터이며 총 60,000개의 훈련 데이터와 레이블, 총 10,000개의 테스트 데이터와 레이블로 구성되어져 있습니다. 레이블은 0부터 9까지 총 10개이며 이 예제는 머신 러닝을 처음 배울 때 접하게 되는 가장 기본적인 예제이기도 합니다. 
  ### Fashoin_Mnist
 
  <p align="center"><img width="900" alt="image" src="https://user-images.githubusercontent.com/97882448/209575744-fd50e361-5fd8-468c-ac03-7cec14ecade2.png">

Fashoin_Mnist는 기존의 MNIST 데이터셋(10개 카테고리의 손으로 쓴 숫자)을 대신해 사용할 수 있는 데이터입니다. MNIST와 동일한 이미지 크기(28x28)이며 훈련데이터와 학습데이터의 수도 동일합니다. 대신 라벨이 숫자가 아니라 옷이며 0부터 9까지 이름은 0:티셔츠/탑, 1: 바지, 2: 풀오버(스웨터의 일종) ,3: 드레스, 4: 코트, 5: 샌들, 6: 셔츠, 7: 스니커즈, 8: 가방, 9:앵클 부츠 입니다.
    

## 코드구현-laddder_networks 모델생성

```python
 #모델을 만들기위해 keras를 불러옴
import keras
from keras.models import *
from keras.layers import *

import tensorflow as tf

#각 레이어마다 베타값을 생성해주는 함수
class AddBeta(Layer):
    def __init__(self  , **kwargs):
        super(AddBeta, self).__init__(**kwargs)
        
    def build(self, input_shape):
        if self.built:
            return
        
        self.beta = self.add_weight(name='beta', 
                                      shape= input_shape[1:] ,
                                      initializer='zeros',
                                      trainable=True)
       
        self.built = True
        super(AddBeta, self).build(input_shape)  
        
    def call(self, x, training=None):
        return tf.add(x, self.beta)

#인코딩에 대한 값을 디코딩해주기 위한 함수
class G_Guass(Layer):
    def __init__(self , **kwargs):
        super(G_Guass, self).__init__(**kwargs)
        
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
            raise ValueError("Invalid argument '%d' provided for init in G_Gauss layer" % init)

#레이어에 훈련 가능한 가중치 변수를 생성하는 함수
    def build(self, input_shape):

        self.size = input_shape[0][-1]

        init_values = [0., 1., 0., 0., 0., 0., 1., 0., 0., 0.]
        self.a = [self.wi(v, 'a' + str(i + 1)) for i, v in enumerate(init_values)]
        super(G_Guass , self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        z_c, u = x 

        def compute(y):
            return y[0] * tf.sigmoid(y[1] * u + y[2]) + y[3] * u + y[4]

        mu = compute(self.a[:5])
        v  = compute(self.a[5:])

        z_est = (z_c - mu) * v + mu
        return z_est
    
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
def get_ladder_network_fc(layer_sizes=[784, 1000, 500, 250, 250, 250, 10], 
     noise_std=0.3,
     denoising_cost=[1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10]):
    #레이어의 갯수 생성
    L = len(layer_sizes) - 1

    inputs_l = Input((layer_sizes[0],))  
    inputs_u = Input((layer_sizes[0],))  

    fc_enc = [Dense(s, use_bias=False, kernel_initializer='glorot_normal') for s in layer_sizes[1:] ]
    fc_dec = [Dense(s, use_bias=False, kernel_initializer='glorot_normal') for s in layer_sizes[:-1]]
    betas  = [AddBeta() for l in range(L)]
    #인코더 정의
    def encoder(inputs, noise_std  ):
        h = add_noise(inputs, noise_std)
        all_z    = [None for _ in range( len(layer_sizes))]
        all_z[0] = h
        
        for l in range(1, L+1):
            z_pre = fc_enc[l-1](h)
            z =     Lambda(batch_normalization)(z_pre) 
            z =     add_noise (z, noise_std)
            
            if l == L:
                h = Activation('softmax')(betas[l-1](z))
            else:
                h = Activation('relu')(betas[l-1](z))
                
            all_z[l] = z

        return h, all_z

    y_c_l, _ = encoder(inputs_l, noise_std)
    y_l, _   = encoder(inputs_l, 0.0)  

    y_c_u, corr_z  = encoder(inputs_u , noise_std)
    y_u,  clean_z  = encoder(inputs_u , 0.0)  

    # 디코더 정의 및  코스트 저장 리스트 생성
    d_cost = []  # to store the denoising cost of all layers
    for l in range(L, -1, -1):
        z, z_c = clean_z[l], corr_z[l]
        if l == L:
            u = y_c_u
        else:
            u = fc_dec[l]( z_est ) 
        u = Lambda(batch_normalization)(u)
        z_est  = G_Guass()([z_c, u])  
        d_cost.append((tf.reduce_mean(tf.reduce_sum(tf.square(z_est - z), 1)) / layer_sizes[l]) * denoising_cost[l])

    u_cost = tf.add_n(d_cost)

    y_c_l = Lambda(lambda x: x[0])([y_c_l, y_l, y_c_u, y_u, u, z_est, z])

    tr_m = Model([inputs_l, inputs_u], y_c_l)
    tr_m.add_loss(u_cost)
    # adam 활용해 최적화 시키고
    tr_m.compile(keras.optimizers.Adam(lr=0.02 ), 'categorical_crossentropy', metrics=['accuracy'])
    #dense loss추출하도록 함
    tr_m.metrics_names.append("final_loss")
    tr_m.metrics_tensors.append(u_cost)
    te_m = Model(inputs_l, y_l)
    tr_m.test_model = te_m

    return tr_m
 ```
## 코드구현-실행파일
 ```python
#from keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
import keras
import random
from sklearn.metrics import accuracy_score
import numpy as np
from ladder_networks import ladder_network

# mnist 데이터의 사이즈
data_size = 28*28
#y값의 갯수
labels = 10
#데이터를 불러옴
(x_train, y_train), (x_test, y_test) =  fashion_mnist.load_data()
#28*28을 1*784로 만들어주고 255를 나눠서 데이터를 정규화하기위한 방법
x_train = x_train.reshape(60000, data_size).astype('float32')/255
x_test  = x_test.reshape(10000,  data_size).astype('float32')/255
#카태고리컬 하게 라벨들을 바꾸어줌 one-hot 인코딩임
y_train = keras.utils.to_categorical(y_train, labels)
y_test  = keras.utils.to_categorical(y_test,  labels)

# 100개 데이터 랜덤추출
numbers_shape = range(x_train.shape[0])
random.seed(2023)
numbers_shape = np.random.choice(x_train.shape[0], 100)

Unlabeled_x_train = x_train
labeled_x_train = x_train[numbers_shape]
labeled_y_train = y_train[numbers_shape]
#몫으로 나눈다음 concat해서 traindata 갯수 맞춰주기
quont = Unlabeled_x_train.shape[0] // labeled_x_train.shape[0]
labeled_x_train_quont = np.concatenate([labeled_x_train]*quont)
labeled_y_train_quont = np.concatenate([labeled_y_train]*quont)

# 모델생성
model = get_ladder_network_fc(layer_sizes=[data_size, 1000, 500, 250, 250, 250, labels])

# 배치사이즈를 100번 돌림
for _ in range(100):
    model.fit([labeled_x_train_quont, Unlabeled_x_train], labeled_y_train_quont, epochs=100)
    y_test_pr = model.test_model.predict(x_test, batch_size=100)
    print("Test 정확도는 %f 나왔습니다." % accuracy_score(y_test.argmax(-1), y_test_pr.argmax(-1)))
 ```
##결론
 

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
Hierarchical latent variable model의 계층적인 특징을 반영한 오토 인코더를 사용하였고 층들의 연결을 Deterministic이 아니라 Stochastic으로 바꿔 주는것이 특징이며 효과적인 학습을 위해 Denoising기법을 활용한 모양이 꼭 사다리와 닮아 있어 Ladder network라고 합니다. Denoising이란 잡음을 추가한 데이터를 학습하여 데이터가 가지고 있는 본래의 고유한 특징을 더 잘 찾기 위한 방법입니다. 구조를 조금더 자세히 보면 먼저 라벨이 있는 데이터는 가우시안 노이즈를 넣은 corrupted  f path와 Clean f path의 값이 빨간색처럼 유사하게 학습을합니다. 또한 corrupted f path가 Denoising q path로 넘어가서 하늘색처럼 Clean f path와  hidden state가 유사하도록 학습합니다. 라벨이 없는 데이터는 답이 없기 때문에 하늘색과 같은 loss만 최소화 시키는 것으로 학습이 진행되고 이러한 구조를 변형해서 디코더에서 가장 높은 layer만  사용하는 모델인 𝛾−모델이 있습니다
  
  

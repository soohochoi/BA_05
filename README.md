# BA_05 Semi-supervised Learning

<p align="center"><img width="604" alt="image" src="https://user-images.githubusercontent.com/97882448/208838979-8e8275d1-1cba-46e7-a584-da075e5a9c54.png">

이자료는 고려대학교 비니지스 애널리틱스 강필성교수님께 배운 Semi-supervised learning을 바탕으로 만들어졌습니다.
먼저, Semi-supervised learning의 방식의 기본적인 개념을 배운후에 Consistency regularization개념 및 코드를 통해 직접 구현을 해봄으로써 이해를 돕도록하겠습니다.

## BA_05 Semi-supervised learning 개념설명

<p align="center"><img width="600" alt="image" src="https://user-images.githubusercontent.com/97882448/209476685-f981bafd-fc68-40a5-9eb9-3460d7e6f185.png">

위 그림처럼 현실에서 데이터에 대한 답이 있는 경우를 예측할 때 supervised learning이라고 하고 답이 없는 경우는 unsupervised learning이라고 합니다. 그러나 우리가 살아가는 세계에는 데이터의 라벨이 조금밖에 없는경우도 다수로 존재합니다. 이럴때 라벨이 조금만 없으면 그것을 제외시켜도 되지만 다수인경우 데이터를 버리기에는 데이터가 아깝습니다. 또한 위에서 오른쪽 그림을 보시면 데이터를 사용해서 분류성능을 더욱 높일 수 있습니다. 단 모든경우에 그렇지는 않기 때문에 분류성능을 높이기 위해서 Pseudo-labeling methods와  Consistency regularization, Hybrid methods와 같은 방법론들이 나오고 있습니다.




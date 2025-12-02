import requests


url = "<YOUR_SERVER_URL>/predict"

data = {
    'items': [{
        "id": 1,
        "title": "Evolutionary Optimization of Physics-Informed Neural Networks: Evo-PINN Frontiers and Opportunities",
        "text": "Deep learning models trained on finite data lack a complete understanding of the physical world. On the other hand, physics-informed neural networks (PINNs) are infused with such knowledge through the incorporation of mathematically expressible laws of nature into their training loss function. By complying with physical laws, PINNs provide advantages over purely data-driven models in limited-data regimes and present as a promising route towards Physical AI. This feature has propelled them to the forefront of scientific machine learning, a domain characterized by scarce and costly data. However, the vision of accurate physics-informed learning comes with significant challenges. This work examines PINNs in terms of model optimization and generalization, shedding light on the need for new algorithmic advances to overcome issues pertaining to the training speed, precision, and generalizability of today’s PINN models. Of particular interest are gradient-free evolutionary algorithms (EAs) for optimizing the uniquely complex loss landscapes arising in PINN training. Methods synergizing gradient descent and EAs for discovering bespoke neural architectures and balancing multiple terms in physics-informed learning objectives are positioned as important avenues for future research. Another exciting track is to cast EAs as a meta-learner of generalizable PINN models. To substantiate these proposed avenues, we further highlight results from recent literature to showcase the early success of such approaches in addressing the aforementioned challenges in PINN optimization and generalization.",
        "mode": 'strict'
    }]
}

# Result
# {"predictions":[{"id":1,"probability":0.13436056673526764,"label":false}]}

r = requests.post(url, json=data)
print(r.text)

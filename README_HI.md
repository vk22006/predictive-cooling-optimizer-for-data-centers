# डेटा केंद्रों के लिए पूर्वानुमानित शीतलन अनुकूलन: ऊर्जा खपत कम करने के लिए तापमान-आधारित चिलर शेड्यूलिंग

[English](README.md) | [தமிழ்](README_TA.md) | [中文](README_ZH.md) | हिन्दी | [Bahasa Indonesia](README_ID.md)

![GitHub top language](https://img.shields.io/github/languages/top/vk22006/predictive-cooling-optimizer-for-data-centers)
![GitHub language count](https://img.shields.io/github/languages/count/vk22006/predictive-cooling-optimizer-for-data-centers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![GitHub forks](https://img.shields.io/github/forks/vk22006/predictive-cooling-optimizer-for-data-centers)

यह परियोजना डेटा केंद्रों की शीतलन प्रणालियों में ऊर्जा की अक्षमता को दूर करने के लिए विकसित की गई है। इसमें तापमान-आधारित पूर्वानुमान मॉडल विकसित किया गया है, जो चिलर के संचालन और शेड्यूलिंग को अनुकूलित करता है। इसका उद्देश्य तापीय सुरक्षा बनाए रखते हुए ऊर्जा खपत को कम करना है। पारंपरिक प्रतिक्रियाशील शीतलन प्रणालियाँ तापमान में परिवर्तन होने के बाद ही प्रतिक्रिया देती हैं, जिसके कारण ऊर्जा की बर्बादी होती है और चिलर का संचालन कम प्रभावी हो जाता है।

![होम पेज](img/home_page.PNG "होम पेज")

## परियोजना की कार्यप्रणाली

इस कार्यप्रणाली की शुरुआत 13,615 HVAC नमूनों के व्यापक डेटा प्रीप्रोसेसिंग से हुई। इसमें IQR का उपयोग करके आउट्लायर का पता लगाना, MinMaxScaler के माध्यम से सामान्यीकरण करना और डेटा की समय-संबंधी अखंडता बनाए रखने के लिए कालानुक्रमिक 80-20 प्रशिक्षण तथा परीक्षण डेटा विभाजन शामिल था।

Feature Engineering के माध्यम से कुल 46 उन्नत फीचर्स बनाए गए। इनमें 16 Lag Features, 12 Rolling Average Features, 6 Cyclical Temporal Encoding Features और 4 Interaction Features शामिल हैं। इनका उपयोग सिस्टम की जटिल गतिशीलताओं को बेहतर ढंग से पकड़ने के लिए किया गया।

दो XGBoost Regression Models ने मुख्य पूर्वानुमान इंजन के रूप में कार्य किया:

* **Energy Prediction Model** ने R² = 0.9891 और MAE = 1.222 kWh प्राप्त किया।
* **Temperature Forecasting Model** ने R² = 0.6853 प्राप्त किया, जिसमें 89.24% पूर्वानुमान ±1°C की सहनशीलता सीमा के भीतर रहे।

दोनों मॉडलों ने कुशल प्रशिक्षण समय प्रदर्शित किया। Energy Prediction Model का प्रशिक्षण समय 2.12 सेकंड और Temperature Forecasting Model का प्रशिक्षण समय 1.87 सेकंड था। इसलिए, ये मॉडल वास्तविक समय में परिनियोजन के लिए उपयुक्त हैं।

`PredictiveCoolingOptimizer` class दोनों मॉडलों को एकीकृत करती है। यह तापमान प्रबंधन के लिए constraint-based रणनीतियों तथा ऊर्जा न्यूनकरण रणनीतियों के माध्यम से पूरे सिस्टम के शीतलन संचालन को अनुकूलित करने में सक्षम बनाती है।

## परीक्षण

कुल 11 परीक्षण पाँच श्रेणियों में किए गए। उनका विवरण नीचे दिया गया है:

|       परीक्षण      |                        लक्ष्य                        |   स्थिति   |
| :---------------: | :-----------------------------------------------: | :-------: |
| Unit Tests        | Energy & Temperature Models, Optimization Engine  |  ✅ सफल |
| Integration Tests | End-to-End Pipeline, System Integration           |  ✅ सफल |
| Functional Tests  | Accuracy, Response Time & Logic                   |  ✅ सफल |
| White Box Test    | Hyperparameters, Feature Engineering              |  ✅ सफल |
| Black Box Test    | Boundary Values, Output Consistency               |  ✅ सफल |
|                   | सफल परीक्षण                                        |  11/11    |
|                   | असफल परीक्षण                                      |  0/11     |
|                   | सफलता दर                                         | 100.0%    |

## निष्पादन की प्रक्रिया

प्रोग्राम को चलाने की प्रक्रिया सरल है। इसे चलाने के लिए निम्न चरणों का पालन करें।

1. आवश्यक libraries इंस्टॉल करें:

```bash
pip install xgboost streamlit
```

2. Command Prompt या PowerShell में परियोजना फ़ोल्डर में जाएँ:

```bash
cd <your-file-path>
```

3. निम्न कमांड का उपयोग करके एप्लिकेशन चलाएँ:

```bash
streamlit run 1_Home.py
```

## उपयोग किए गए टूल्स

1. Anaconda Jupyter - मॉडल प्रशिक्षण और परीक्षण के लिए
2. Streamlit Library - फ्रंटएंड कार्यान्वयन के लिए
3. Joblib - `.pkl` मॉडल फ़ाइलों को संभालने के लिए
4. NumPy
5. Pandas
6. Scikit-Learn
7. XGBoost

## उपयोग किए गए एल्गोरिदम

### 1. पूर्वानुमान एल्गोरिदम

* XGBoost (Extreme Gradient Boosting)
* Random Forest Regressor

### 2. सहायक एल्गोरिदम

* Min-Max Normalization
* Rolling Average (Feature Engineering के लिए)

यह टूल सॉफ्टवेयर-आधारित पूर्वानुमानित शीतलन अनुकूलन की व्यवहार्यता को सफलतापूर्वक प्रदर्शित करता है। प्रशिक्षित मॉडल Streamlit का उपयोग करके बनाए गए इंटरैक्टिव वेब एप्लिकेशन में परिनियोजित किए जाने के लिए तैयार हैं। इससे उपयोगकर्ता-अनुकूल इंटरफ़ेस, सिस्टम प्रदर्शन और संबंधित हितधारकों के समक्ष परियोजना के प्रदर्शन को संभव बनाया जा सकता है।

## लाइसेंस

यह परियोजना MIT License के अंतर्गत लाइसेंस प्राप्त है। अधिक जानकारी के लिए [LICENSE](LICENSE) फ़ाइल देखें।

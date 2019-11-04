# Kaggle IEEE-CIS-Fraud-Detection

竞赛由IEEE-CIS协会联合Vesta Corporation共同举办，旨在提升金融防欺诈算法准确性，详细信息请访问[官方网址](https://www.kaggle.com/c/ieee-fraud-detection/overview)。这个赛题特点：**二分类**，**不均衡样本**，**时间序列数据**，**半匿名特征**。

## 数据信息

官方提供交易数据表（Transaction）和属性数据表（Identity）。

### 交易表

- TransactionDT 相对时间戳

- TransactionAMT 交易额（$）

- ProductCD 交易产品编码

- card1 - card6 支付卡类型，卡类别，发行银行，国家/地区等

- addr 地址

- dist  距离

- P_ and (R__) emaildomain 购买者和接收者邮箱域名

- C1-C14 各类计数，例如发现多少地址与支付卡相关联，实际含义被掩盖

- D1-D15 时间信息，例如距离上次交易时间多少天

- M1-M9 匹配，例如信用卡名字与地址是否匹配

- V1-V300 Vesta官方开发的特征，包括排名、计数、实体间关系等

其中类别型特征为：ProductCD、card1-card6、addr1-addr2、emaildomin、M1-M9

### 属性表

该表中的变量是与交易相关的身份信息 - 网络连接信息（IP，ISP，代理等）和数字签名（UA /浏览器/操作系统/版本等）。它们由Vesta的欺诈保护系统和数字安全合作伙伴收集，字段名称被屏蔽。

类别型特征为：DeviceType、DeviceInfo、id*12 - id*38

### 讨论区总结

**[数据描述讨论](https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203)**

- 交易表 “它包含汇款以及其他礼品和服务，例如你为其他人预订了机票等。”
- TransactionDT第一个值是86400，它对应于一天中的秒数（60 * 60 * 24 = 86400）所以我认为单位是秒。使用这个，我们知道数据跨越6个月，因为最大值是15811131，这对应于第183天。“
- TransactionAMT：“某些交易金额在小数点右侧有三位小数。似乎有一个指向三个小数位的链接以及一个空白的addr1和addr2字段。是否有可能这些是外国交易，例如，第12行中的75.887是外币数量乘以汇率的结果？“
- productCD：产品不一定是真正的'产品'（比如要添加到购物车中的一件商品）。它可以是任何一种服务。
- addr：地址是针对购买者的，addr1是购买区域，addr是购买国家/地区；
- dist：（不限于）帐单邮寄地址，邮寄地址，邮政编码，IP地址，电话区域等之间的距离
- emaildomain：某些交易不需要收件人，因此Remaildomain为空。
- C1-C14：客户的电话数量、邮箱数量、地址数量、设备数量等。
- V1-V300：例如，与IP和电子邮件或地址关联的支付卡在24小时时间范围内出现的次数等，所有的V特征都是数值方法计算得到的，其中一些是聚类中的订单数量，时间段或条件，因此值是有限的并且具有排序（或排名）。大部分人更倾向于这些不是类别型特征，除非刚好出现只有2个值的情况，也许可以视作类别型特征。
- id：id01到id11是身份的数字特征，由Vesta和安全合作伙伴收集，例如设备评级，ip_domain评级，代理评级等。此外，它还记录了行为指纹，如帐户登录时间/登录时间失败，帐户在页面上停留的时间等。所有这些都无法详细说明。我希望你能够得到这些特征的基本含义，并且通过提及它们是数字/分类，你不会不恰当地处理它们。

**[数据提供商回复](https://www.kaggle.com/c/ieee-fraud-detection/discussion/100304#latest-590684)**

Q: How did you label the target? Is that way(answer of above question) totally same as both train and test? e.g. timespan for figuring out if it's fraud.

A：We define reported chargeback on card, user account, associated email address and transactions directly linked to these attributes as fraud transaction (isFraud=1); If none of above is found after 120 days, then we define as legit (isFraud=0). This is applied to both train and test. The date time for provided dataset is long enough from now to believe that label is reliable.

Q: Are category variables ordered or unordered categorical variables? 

A: All variables marked as categorical are unordered. There're some of boolean variables marked as categorical too. The remaining variables are having numerical characteristics (one of which is the ordering matters). Your observation is correct - there're some numerical variables are having limited values. For example, how many email address are linked to a payment card; the count is an integer, which you can treat as either numerical or categorical; However, ordering of that indeed indicates risk level.

Q: Any counties informations?

A: These payments are from different countries, including North America, Latin America, Europe. But TransactionAmt has been converted to USD.

Q: If a card was used in a fraudulent transaction, will every transaction that uses this card in the future be flagged as a fraud?

A: Yes, all transactions linked to the same payment card which is identified as fraudulent will be labeled as fraud. However, exact card information is NOT included in the dataset.

**[Must-see kernels for 'IEEE-CIS Fraud Detection']('IEEE-CIS Fraud Detection')**

[**Feature Engineering: Time of day**](https://www.kaggle.com/c/ieee-fraud-detection/discussion/100400#latest-616798)

- Chris Deotte :X*train['D9'] = (X*train['TransactionDT']%(3600*24)/3600//1)/24.0.
  Some people's models directly put time features into the model will have a lower score, but some time into statistical features have increased.have fun
- The hour of the day is D9.

## 数据处理

- 高缺失
- 高重复
- 类别型变量转换
- 时间
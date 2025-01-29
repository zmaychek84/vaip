<!--
    Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
    Licensed under the MIT License.
 -->

# pass dd merge qmhagrpb


## pattern in mxgan


```mermaid
flowchart TB
    com_microsoft_DequantizeLinear_0["com_microsoft_DequantizeLinear_0<br>[], ty=1<br>107_DequantizeLinear_Output/duplicated"]
    classDef com_microsoft_DequantizeLinear fill:#ffffe0,stroke:#ff8c00,stroke-width:2px;;
    class com_microsoft_DequantizeLinear_0 com_microsoft_DequantizeLinear;
    input_0[\"input_0<br>[1,512,1152], ty=2<br>274_QuantizeLinear_Output"/]
    classDef input fill:#add8e6,stroke:#00008b,stroke-width:2px;;
    class input_0 input;
    com_microsoft_DequantizeLinear_1["com_microsoft_DequantizeLinear_1<br>[1,512,1152], ty=1<br>274_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_1 com_microsoft_DequantizeLinear;
    input_0 -.-> com_microsoft_DequantizeLinear_1
    Reshape_0["Reshape_0<br>[1,512,12,96], ty=1<br>297"]
    classDef Reshape fill:#e6e6fa,stroke:#9400d3,stroke-width:2px;;
    class Reshape_0 Reshape;
    com_microsoft_DequantizeLinear_1 -.-> Reshape_0
    com_microsoft_QuantizeLinear_0["com_microsoft_QuantizeLinear_0<br>[1,512,12,96], ty=2<br>297_QuantizeLinear_Output"]
    classDef com_microsoft_QuantizeLinear fill:#f08080,stroke:#8b0000,stroke-width:2px;;
    class com_microsoft_QuantizeLinear_0 com_microsoft_QuantizeLinear;
    Reshape_0 -.-> com_microsoft_QuantizeLinear_0
    com_microsoft_DequantizeLinear_2["com_microsoft_DequantizeLinear_2<br>[1,512,12,96], ty=1<br>297_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_2 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_0 -.-> com_microsoft_DequantizeLinear_2
    Transpose_0["Transpose_0<br>[1,12,512,96], ty=1<br>298"]
    classDef Transpose fill:#ffdab9,stroke:#8b4513,stroke-width:2px;;
    class Transpose_0 Transpose;
    com_microsoft_DequantizeLinear_2 -.-> Transpose_0
    com_microsoft_QuantizeLinear_1["com_microsoft_QuantizeLinear_1<br>[1,12,512,96], ty=2<br>298_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_1 com_microsoft_QuantizeLinear;
    Transpose_0 -.-> com_microsoft_QuantizeLinear_1
    com_microsoft_DequantizeLinear_3["com_microsoft_DequantizeLinear_3<br>[1,12,512,96], ty=1<br>298_DequantizeLinear_Output/duplicated"]
    class com_microsoft_DequantizeLinear_3 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_1 -.-> com_microsoft_DequantizeLinear_3
    com_microsoft_DequantizeLinear_4["com_microsoft_DequantizeLinear_4<br>[96,8], ty=1<br>1077_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_4 com_microsoft_DequantizeLinear;
    MatMul_0["MatMul_0<br>[1,12,512,8], ty=1<br>351"]
    classDef MatMul fill:#e0ffff,stroke:#008080,stroke-width:2px;;
    class MatMul_0 MatMul;
    com_microsoft_DequantizeLinear_3 -.-> MatMul_0
    com_microsoft_DequantizeLinear_4 -.-> MatMul_0
    com_microsoft_QuantizeLinear_2["com_microsoft_QuantizeLinear_2<br>[1,12,512,8], ty=2<br>351_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_2 com_microsoft_QuantizeLinear;
    MatMul_0 -.-> com_microsoft_QuantizeLinear_2
    com_microsoft_DequantizeLinear_5["com_microsoft_DequantizeLinear_5<br>[1,12,512,8], ty=1<br>351_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_5 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_2 -.-> com_microsoft_DequantizeLinear_5
    com_microsoft_DequantizeLinear_6["com_microsoft_DequantizeLinear_6<br>[8], ty=1<br>roberta_encoder_src.encoder.layer.0.attention.self.gate_ur_linear.bias_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_6 com_microsoft_DequantizeLinear;
    Add_0["Add_0<br>[1,12,512,8], ty=1<br>352"]
    classDef Add fill:#f5f5dc,stroke:#8b4513,stroke-width:2px;;
    class Add_0 Add;
    com_microsoft_DequantizeLinear_5 -.-> Add_0
    com_microsoft_DequantizeLinear_6 -.-> Add_0
    com_microsoft_QuantizeLinear_3["com_microsoft_QuantizeLinear_3<br>[1,12,512,8], ty=2<br>352_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_3 com_microsoft_QuantizeLinear;
    Add_0 -.-> com_microsoft_QuantizeLinear_3
    com_microsoft_DequantizeLinear_7["com_microsoft_DequantizeLinear_7<br>[1,12,512,8], ty=1<br>352_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_7 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_3 -.-> com_microsoft_DequantizeLinear_7
    Reshape_1["Reshape_1<br>[1,12,512,2,4], ty=1<br>366"]
    class Reshape_1 Reshape;
    com_microsoft_DequantizeLinear_7 -.-> Reshape_1
    com_microsoft_QuantizeLinear_4["com_microsoft_QuantizeLinear_4<br>[1,12,512,2,4], ty=2<br>366_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_4 com_microsoft_QuantizeLinear;
    Reshape_1 -.-> com_microsoft_QuantizeLinear_4
    com_microsoft_DequantizeLinear_8["com_microsoft_DequantizeLinear_8<br>[1,12,512,2,4], ty=1<br>366_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_8 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_4 -.-> com_microsoft_DequantizeLinear_8
    ReduceSum_0["ReduceSum_0<br>[1,12,512,2], ty=1<br>368"]
    com_microsoft_DequantizeLinear_8 -.-> ReduceSum_0
    com_microsoft_QuantizeLinear_5["com_microsoft_QuantizeLinear_5<br>[1,12,512,2], ty=2<br>368_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_5 com_microsoft_QuantizeLinear;
    ReduceSum_0 -.-> com_microsoft_QuantizeLinear_5
    com_microsoft_DequantizeLinear_9["com_microsoft_DequantizeLinear_9<br>[1,12,512,2], ty=1<br>368_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_9 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_5 -.-> com_microsoft_DequantizeLinear_9
    Sigmoid_0["Sigmoid_0<br>[1,12,512,2], ty=1<br>369"]
    classDef Sigmoid fill:#d3d3d3,stroke:#36454f,stroke-width:2px;;
    class Sigmoid_0 Sigmoid;
    com_microsoft_DequantizeLinear_9 -.-> Sigmoid_0
    com_microsoft_QuantizeLinear_6["com_microsoft_QuantizeLinear_6<br>[1,12,512,2], ty=2<br>369_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_6 com_microsoft_QuantizeLinear;
    Sigmoid_0 -.-> com_microsoft_QuantizeLinear_6
    com_microsoft_DequantizeLinear_10["com_microsoft_DequantizeLinear_10<br>[1,12,512,2], ty=1<br>369_DequantizeLinear_Output/duplicated"]
    class com_microsoft_DequantizeLinear_10 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_6 -.-> com_microsoft_DequantizeLinear_10
    Slice_0["Slice_0<br>[1,12,512,1], ty=1<br>383"]
    com_microsoft_DequantizeLinear_10 -.-> Slice_0
    com_microsoft_QuantizeLinear_7["com_microsoft_QuantizeLinear_7<br>[1,12,512,1], ty=2<br>383_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_7 com_microsoft_QuantizeLinear;
    Slice_0 -.-> com_microsoft_QuantizeLinear_7
    com_microsoft_DequantizeLinear_11["com_microsoft_DequantizeLinear_11<br>[1,12,512,1], ty=1<br>383_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_11 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_7 -.-> com_microsoft_DequantizeLinear_11
    com_microsoft_DequantizeLinear_12["com_microsoft_DequantizeLinear_12<br>[1,12,1,1], ty=1<br>roberta_encoder_src.encoder.layer.0.attention.self.eco_a_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_12 com_microsoft_DequantizeLinear;
    Mul_0["Mul_0<br>[1,12,512,1], ty=1<br>384"]
    com_microsoft_DequantizeLinear_11 -.-> Mul_0
    com_microsoft_DequantizeLinear_12 -.-> Mul_0
    com_microsoft_QuantizeLinear_8["com_microsoft_QuantizeLinear_8<br>[1,12,512,1], ty=2<br>384_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_8 com_microsoft_QuantizeLinear;
    Mul_0 -.-> com_microsoft_QuantizeLinear_8
    com_microsoft_DequantizeLinear_13["com_microsoft_DequantizeLinear_13<br>[1,12,512,1], ty=1<br>384_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_13 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_8 -.-> com_microsoft_DequantizeLinear_13
    Sub_0["Sub_0<br>[1,12,512,1], ty=1<br>386"]
    classDef Sub fill:#add8e6,stroke:#00008b,stroke-width:2px;;
    class Sub_0 Sub;
    com_microsoft_DequantizeLinear_13 -.-> Sub_0
    com_microsoft_DequantizeLinear_0 -.-> Sub_0
    com_microsoft_QuantizeLinear_9["com_microsoft_QuantizeLinear_9<br>[1,12,512,1], ty=2<br>386_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_9 com_microsoft_QuantizeLinear;
    Sub_0 -.-> com_microsoft_QuantizeLinear_9
    com_microsoft_DequantizeLinear_14["com_microsoft_DequantizeLinear_14<br>[1,12,512,1], ty=1<br>386_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_14 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_9 -.-> com_microsoft_DequantizeLinear_14
    com_microsoft_DequantizeLinear_15["com_microsoft_DequantizeLinear_15<br>[1,12,512,2], ty=1<br>369_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_15 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_6 -.-> com_microsoft_DequantizeLinear_15
    Slice_1["Slice_1<br>[1,12,512,1], ty=1<br>380"]
    com_microsoft_DequantizeLinear_15 -.-> Slice_1
    com_microsoft_QuantizeLinear_10["com_microsoft_QuantizeLinear_10<br>[1,12,512,1], ty=2<br>380_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_10 com_microsoft_QuantizeLinear;
    Slice_1 -.-> com_microsoft_QuantizeLinear_10
    com_microsoft_DequantizeLinear_16["com_microsoft_DequantizeLinear_16<br>[1,12,512,1], ty=1<br>380_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_16 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_10 -.-> com_microsoft_DequantizeLinear_16
    Mul_1["Mul_1<br>[1,12,512,1], ty=1<br>387"]
    com_microsoft_DequantizeLinear_16 -.-> Mul_1
    com_microsoft_DequantizeLinear_14 -.-> Mul_1
    com_microsoft_QuantizeLinear_11["com_microsoft_QuantizeLinear_11<br>[1,12,512,1], ty=2<br>387_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_11 com_microsoft_QuantizeLinear;
    Mul_1 -.-> com_microsoft_QuantizeLinear_11
    com_microsoft_DequantizeLinear_17["com_microsoft_DequantizeLinear_17<br>[1,12,512,1], ty=1<br>387_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_17 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_11 -.-> com_microsoft_DequantizeLinear_17
    com_microsoft_DequantizeLinear_18["com_microsoft_DequantizeLinear_18<br>[], ty=1<br>130_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_18 com_microsoft_DequantizeLinear;
    Add_1["Add_1<br>[1,12,512,1], ty=1<br>389"]
    class Add_1 Add;
    com_microsoft_DequantizeLinear_17 -.-> Add_1
    com_microsoft_DequantizeLinear_18 -.-> Add_1
    com_microsoft_QuantizeLinear_12["com_microsoft_QuantizeLinear_12<br>[1,12,512,1], ty=2<br>389_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_12 com_microsoft_QuantizeLinear;
    Add_1 -.-> com_microsoft_QuantizeLinear_12
    com_microsoft_DequantizeLinear_19["com_microsoft_DequantizeLinear_19<br>[1,12,512,1], ty=1<br>389_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_19 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_12 -.-> com_microsoft_DequantizeLinear_19
    com_microsoft_DequantizeLinear_20["com_microsoft_DequantizeLinear_20<br>[1,12,512,512], ty=1<br>271_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_20 com_microsoft_DequantizeLinear;
    Mul_2["Mul_2<br>[1,12,512,512], ty=1<br>390"]
    com_microsoft_DequantizeLinear_19 -.-> Mul_2
    com_microsoft_DequantizeLinear_20 -.-> Mul_2
    com_microsoft_QuantizeLinear_13["com_microsoft_QuantizeLinear_13<br>[1,12,512,512], ty=2<br>390_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_13 com_microsoft_QuantizeLinear;
    Mul_2 -.-> com_microsoft_QuantizeLinear_13
    com_microsoft_DequantizeLinear_21["com_microsoft_DequantizeLinear_21<br>[1,12,512,512], ty=1<br>390_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_21 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_13 -.-> com_microsoft_DequantizeLinear_21
    com_microsoft_QuantizeLinear_14["com_microsoft_QuantizeLinear_14<br>[1,12,512,512], ty=4<br>390_convert_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_14 com_microsoft_QuantizeLinear;
    com_microsoft_DequantizeLinear_21 -.-> com_microsoft_QuantizeLinear_14
    com_microsoft_DequantizeLinear_22["com_microsoft_DequantizeLinear_22<br>[1,12,512,512], ty=1<br>390_convert_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_22 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_14 -.-> com_microsoft_DequantizeLinear_22
    com_microsoft_DequantizeLinear_23["com_microsoft_DequantizeLinear_23<br>[1,12,512,96], ty=1<br>298_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_23 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_1 -.-> com_microsoft_DequantizeLinear_23
    input_1[\"input_1<br>[1,512,1152], ty=2<br>276_QuantizeLinear_Output"/]
    class input_1 input;
    com_microsoft_DequantizeLinear_24["com_microsoft_DequantizeLinear_24<br>[1,512,1152], ty=1<br>276_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_24 com_microsoft_DequantizeLinear;
    input_1 -.-> com_microsoft_DequantizeLinear_24
    Reshape_2["Reshape_2<br>[1,512,12,96], ty=1<br>316"]
    class Reshape_2 Reshape;
    com_microsoft_DequantizeLinear_24 -.-> Reshape_2
    com_microsoft_QuantizeLinear_15["com_microsoft_QuantizeLinear_15<br>[1,512,12,96], ty=2<br>316_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_15 com_microsoft_QuantizeLinear;
    Reshape_2 -.-> com_microsoft_QuantizeLinear_15
    com_microsoft_DequantizeLinear_25["com_microsoft_DequantizeLinear_25<br>[1,512,12,96], ty=1<br>316_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_25 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_15 -.-> com_microsoft_DequantizeLinear_25
    Transpose_1["Transpose_1<br>[1,12,96,512], ty=1<br>336"]
    class Transpose_1 Transpose;
    com_microsoft_DequantizeLinear_25 -.-> Transpose_1
    com_microsoft_QuantizeLinear_16["com_microsoft_QuantizeLinear_16<br>[1,12,96,512], ty=2<br>336_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_16 com_microsoft_QuantizeLinear;
    Transpose_1 -.-> com_microsoft_QuantizeLinear_16
    com_microsoft_DequantizeLinear_26["com_microsoft_DequantizeLinear_26<br>[1,12,96,512], ty=1<br>336_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_26 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_16 -.-> com_microsoft_DequantizeLinear_26
    MatMul_1["MatMul_1<br>[1,12,512,512], ty=1<br>337"]
    class MatMul_1 MatMul;
    com_microsoft_DequantizeLinear_23 -.-> MatMul_1
    com_microsoft_DequantizeLinear_26 -.-> MatMul_1
    com_microsoft_QuantizeLinear_17["com_microsoft_QuantizeLinear_17<br>[1,12,512,512], ty=2<br>337_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_17 com_microsoft_QuantizeLinear;
    MatMul_1 -.-> com_microsoft_QuantizeLinear_17
    com_microsoft_DequantizeLinear_27["com_microsoft_DequantizeLinear_27<br>[1,12,512,512], ty=1<br>337_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_27 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_17 -.-> com_microsoft_DequantizeLinear_27
    com_microsoft_DequantizeLinear_28["com_microsoft_DequantizeLinear_28<br>[], ty=1<br>1062_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_28 com_microsoft_DequantizeLinear;
    Div_0["Div_0<br>[1,12,512,512], ty=1<br>339"]
    com_microsoft_DequantizeLinear_27 -.-> Div_0
    com_microsoft_DequantizeLinear_28 -.-> Div_0
    com_microsoft_QuantizeLinear_18["com_microsoft_QuantizeLinear_18<br>[1,12,512,512], ty=2<br>339_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_18 com_microsoft_QuantizeLinear;
    Div_0 -.-> com_microsoft_QuantizeLinear_18
    com_microsoft_DequantizeLinear_29["com_microsoft_DequantizeLinear_29<br>[1,12,512,512], ty=1<br>339_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_29 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_18 -.-> com_microsoft_DequantizeLinear_29
    com_microsoft_QuantizeLinear_19["com_microsoft_QuantizeLinear_19<br>[1,12,512,512], ty=4<br>339_convert_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_19 com_microsoft_QuantizeLinear;
    com_microsoft_DequantizeLinear_29 -.-> com_microsoft_QuantizeLinear_19
    com_microsoft_DequantizeLinear_30["com_microsoft_DequantizeLinear_30<br>[1,12,512,512], ty=1<br>339_convert_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_30 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_19 -.-> com_microsoft_DequantizeLinear_30
    input_2[\"input_2<br>[1,1,1,512], ty=4<br>110_convert_QuantizeLinear_Output"/]
    class input_2 input;
    com_microsoft_DequantizeLinear_31["com_microsoft_DequantizeLinear_31<br>[1,1,1,512], ty=1<br>110_convert_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_31 com_microsoft_DequantizeLinear;
    input_2 -.-> com_microsoft_DequantizeLinear_31
    Add_2["Add_2<br>[1,12,512,512], ty=1<br>340"]
    class Add_2 Add;
    com_microsoft_DequantizeLinear_30 -.-> Add_2
    com_microsoft_DequantizeLinear_31 -.-> Add_2
    com_microsoft_QuantizeLinear_20["com_microsoft_QuantizeLinear_20<br>[1,12,512,512], ty=4<br>340_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_20 com_microsoft_QuantizeLinear;
    Add_2 -.-> com_microsoft_QuantizeLinear_20
    com_microsoft_DequantizeLinear_32["com_microsoft_DequantizeLinear_32<br>[1,12,512,512], ty=1<br>340_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_32 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_20 -.-> com_microsoft_DequantizeLinear_32
    Add_3["Add_3<br>[1,12,512,512], ty=1<br>391"]
    class Add_3 Add;
    com_microsoft_DequantizeLinear_32 -.-> Add_3
    com_microsoft_DequantizeLinear_22 -.-> Add_3
    com_microsoft_QuantizeLinear_21["com_microsoft_QuantizeLinear_21<br>[1,12,512,512], ty=4<br>391_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_21 com_microsoft_QuantizeLinear;
    Add_3 -.-> com_microsoft_QuantizeLinear_21
    com_microsoft_DequantizeLinear_33["com_microsoft_DequantizeLinear_33<br>[1,12,512,512], ty=1<br>391_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_33 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_21 -.-> com_microsoft_DequantizeLinear_33
    Softmax_0["Softmax_0<br>[1,12,512,512], ty=1<br>392"]
    com_microsoft_DequantizeLinear_33 -.-> Softmax_0
    com_microsoft_QuantizeLinear_22["com_microsoft_QuantizeLinear_22<br>[1,12,512,512], ty=4<br>392_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_22 com_microsoft_QuantizeLinear;
    Softmax_0 -.-> com_microsoft_QuantizeLinear_22
    com_microsoft_DequantizeLinear_34["com_microsoft_DequantizeLinear_34<br>[1,12,512,512], ty=1<br>392_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_34 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_22 -.-> com_microsoft_DequantizeLinear_34
    com_microsoft_QuantizeLinear_23["com_microsoft_QuantizeLinear_23<br>[1,12,512,512], ty=2<br>392_convert_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_23 com_microsoft_QuantizeLinear;
    com_microsoft_DequantizeLinear_34 -.-> com_microsoft_QuantizeLinear_23
    com_microsoft_DequantizeLinear_35["com_microsoft_DequantizeLinear_35<br>[1,12,512,512], ty=1<br>392_convert_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_35 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_23 -.-> com_microsoft_DequantizeLinear_35
    input_3[\"input_3<br>[1,512,768], ty=2<br>279_QuantizeLinear_Output"/]
    class input_3 input;
    com_microsoft_DequantizeLinear_36["com_microsoft_DequantizeLinear_36<br>[1,512,768], ty=1<br>279_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_36 com_microsoft_DequantizeLinear;
    input_3 -.-> com_microsoft_DequantizeLinear_36
    Reshape_3["Reshape_3<br>[1,512,12,64], ty=1<br>334"]
    class Reshape_3 Reshape;
    com_microsoft_DequantizeLinear_36 -.-> Reshape_3
    com_microsoft_QuantizeLinear_24["com_microsoft_QuantizeLinear_24<br>[1,512,12,64], ty=2<br>334_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_24 com_microsoft_QuantizeLinear;
    Reshape_3 -.-> com_microsoft_QuantizeLinear_24
    com_microsoft_DequantizeLinear_37["com_microsoft_DequantizeLinear_37<br>[1,512,12,64], ty=1<br>334_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_37 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_24 -.-> com_microsoft_DequantizeLinear_37
    Transpose_2["Transpose_2<br>[1,12,512,64], ty=1<br>335"]
    class Transpose_2 Transpose;
    com_microsoft_DequantizeLinear_37 -.-> Transpose_2
    com_microsoft_QuantizeLinear_25["com_microsoft_QuantizeLinear_25<br>[1,12,512,64], ty=2<br>335_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_25 com_microsoft_QuantizeLinear;
    Transpose_2 -.-> com_microsoft_QuantizeLinear_25
    com_microsoft_DequantizeLinear_38["com_microsoft_DequantizeLinear_38<br>[1,12,512,64], ty=1<br>335_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_38 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_25 -.-> com_microsoft_DequantizeLinear_38
    MatMul_2["MatMul_2<br>[1,12,512,64], ty=1<br>393"]
    class MatMul_2 MatMul;
    com_microsoft_DequantizeLinear_35 -.-> MatMul_2
    com_microsoft_DequantizeLinear_38 -.-> MatMul_2
    com_microsoft_QuantizeLinear_26["com_microsoft_QuantizeLinear_26<br>[1,12,512,64], ty=2<br>393_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_26 com_microsoft_QuantizeLinear;
    MatMul_2 -.-> com_microsoft_QuantizeLinear_26
    com_microsoft_DequantizeLinear_39["com_microsoft_DequantizeLinear_39<br>[1,12,512,64], ty=1<br>393_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_39 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_26 -.-> com_microsoft_DequantizeLinear_39
    Transpose_3["Transpose_3<br>[1,512,12,64], ty=1<br>394"]
    class Transpose_3 Transpose;
    com_microsoft_DequantizeLinear_39 -.-> Transpose_3
    com_microsoft_QuantizeLinear_27["com_microsoft_QuantizeLinear_27<br>[1,512,12,64], ty=2<br>394_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_27 com_microsoft_QuantizeLinear;
    Transpose_3 -.-> com_microsoft_QuantizeLinear_27
    com_microsoft_DequantizeLinear_40["com_microsoft_DequantizeLinear_40<br>[1,512,12,64], ty=1<br>394_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_40 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_27 -.-> com_microsoft_DequantizeLinear_40
    Reshape_4["Reshape_4<br>[1,512,768], ty=1<br>409"]
    class Reshape_4 Reshape;
    com_microsoft_DequantizeLinear_40 -.-> Reshape_4
    com_microsoft_QuantizeLinear_28[["com_microsoft_QuantizeLinear_28<br>[1,512,768], ty=2<br>409_QuantizeLinear_Output"]]
    class com_microsoft_QuantizeLinear_28 com_microsoft_QuantizeLinear;
    Reshape_4 --o com_microsoft_QuantizeLinear_28
```

## pattern in mxpzi


```mermaid
flowchart TB
    com_microsoft_DequantizeLinear_0["com_microsoft_DequantizeLinear_0<br>[], ty=1<br>/lang_encoder/Constant_3_output_0_DequantizeLinear_Output/duplicated"]
    classDef com_microsoft_DequantizeLinear fill:#ffffe0,stroke:#ff8c00,stroke-width:2px;;
    class com_microsoft_DequantizeLinear_0 com_microsoft_DequantizeLinear;
    input_0[\"input_0<br>[1,77,1152], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/query/Add_output_0_QuantizeLinear_Output"/]
    classDef input fill:#add8e6,stroke:#00008b,stroke-width:2px;;
    class input_0 input;
    com_microsoft_DequantizeLinear_1["com_microsoft_DequantizeLinear_1<br>[1,77,1152], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/query/Add_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_1 com_microsoft_DequantizeLinear;
    input_0 -.-> com_microsoft_DequantizeLinear_1
    Reshape_0["Reshape_0<br>[1,77,12,96], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Reshape_output_0"]
    classDef Reshape fill:#e6e6fa,stroke:#9400d3,stroke-width:2px;;
    class Reshape_0 Reshape;
    com_microsoft_DequantizeLinear_1 -.-> Reshape_0
    com_microsoft_QuantizeLinear_0["com_microsoft_QuantizeLinear_0<br>[1,77,12,96], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/Reshape_output_0_QuantizeLinear_Output"]
    classDef com_microsoft_QuantizeLinear fill:#f08080,stroke:#8b0000,stroke-width:2px;;
    class com_microsoft_QuantizeLinear_0 com_microsoft_QuantizeLinear;
    Reshape_0 -.-> com_microsoft_QuantizeLinear_0
    com_microsoft_DequantizeLinear_2["com_microsoft_DequantizeLinear_2<br>[1,77,12,96], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Reshape_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_2 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_0 -.-> com_microsoft_DequantizeLinear_2
    Transpose_0["Transpose_0<br>[1,12,77,96], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Transpose_output_0"]
    classDef Transpose fill:#ffdab9,stroke:#8b4513,stroke-width:2px;;
    class Transpose_0 Transpose;
    com_microsoft_DequantizeLinear_2 -.-> Transpose_0
    com_microsoft_QuantizeLinear_1["com_microsoft_QuantizeLinear_1<br>[1,12,77,96], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/Transpose_output_0_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_1 com_microsoft_QuantizeLinear;
    Transpose_0 -.-> com_microsoft_QuantizeLinear_1
    com_microsoft_DequantizeLinear_3["com_microsoft_DequantizeLinear_3<br>[1,12,77,96], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear_Output/duplicated"]
    class com_microsoft_DequantizeLinear_3 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_1 -.-> com_microsoft_DequantizeLinear_3
    com_microsoft_DequantizeLinear_4["com_microsoft_DequantizeLinear_4<br>[96,8], ty=1<br>onnx::MatMul_2457_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_4 com_microsoft_DequantizeLinear;
    MatMul_0["MatMul_0<br>[1,12,77,8], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/gate_ur_linear/MatMul_output_0"]
    classDef MatMul fill:#e0ffff,stroke:#008080,stroke-width:2px;;
    class MatMul_0 MatMul;
    com_microsoft_DequantizeLinear_3 -.-> MatMul_0
    com_microsoft_DequantizeLinear_4 -.-> MatMul_0
    com_microsoft_QuantizeLinear_2["com_microsoft_QuantizeLinear_2<br>[1,12,77,8], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/gate_ur_linear/MatMul_output_0_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_2 com_microsoft_QuantizeLinear;
    MatMul_0 -.-> com_microsoft_QuantizeLinear_2
    com_microsoft_DequantizeLinear_5["com_microsoft_DequantizeLinear_5<br>[1,12,77,8], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/gate_ur_linear/MatMul_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_5 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_2 -.-> com_microsoft_DequantizeLinear_5
    com_microsoft_DequantizeLinear_6["com_microsoft_DequantizeLinear_6<br>[8], ty=1<br>lang_encoder.encoder.layer.0.attention.self.gate_ur_linear.bias_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_6 com_microsoft_DequantizeLinear;
    Add_0["Add_0<br>[1,12,77,8], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/gate_ur_linear/Add_output_0"]
    classDef Add fill:#f5f5dc,stroke:#8b4513,stroke-width:2px;;
    class Add_0 Add;
    com_microsoft_DequantizeLinear_6 -.-> Add_0
    com_microsoft_DequantizeLinear_5 -.-> Add_0
    com_microsoft_QuantizeLinear_3["com_microsoft_QuantizeLinear_3<br>[1,12,77,8], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/gate_ur_linear/Add_output_0_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_3 com_microsoft_QuantizeLinear;
    Add_0 -.-> com_microsoft_QuantizeLinear_3
    com_microsoft_DequantizeLinear_7["com_microsoft_DequantizeLinear_7<br>[1,12,77,8], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/gate_ur_linear/Add_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_7 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_3 -.-> com_microsoft_DequantizeLinear_7
    Reshape_1["Reshape_1<br>[1,12,77,2,4], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Reshape_3_output_0"]
    class Reshape_1 Reshape;
    com_microsoft_DequantizeLinear_7 -.-> Reshape_1
    com_microsoft_QuantizeLinear_4["com_microsoft_QuantizeLinear_4<br>[1,12,77,2,4], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/Reshape_3_output_0_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_4 com_microsoft_QuantizeLinear;
    Reshape_1 -.-> com_microsoft_QuantizeLinear_4
    com_microsoft_DequantizeLinear_8["com_microsoft_DequantizeLinear_8<br>[1,12,77,2,4], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Reshape_3_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_8 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_4 -.-> com_microsoft_DequantizeLinear_8
    ReduceSum_0["ReduceSum_0<br>[1,12,77,2], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/ReduceSum_output_0"]
    com_microsoft_DequantizeLinear_8 -.-> ReduceSum_0
    com_microsoft_QuantizeLinear_5["com_microsoft_QuantizeLinear_5<br>[1,12,77,2], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/ReduceSum_output_0_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_5 com_microsoft_QuantizeLinear;
    ReduceSum_0 -.-> com_microsoft_QuantizeLinear_5
    com_microsoft_DequantizeLinear_9["com_microsoft_DequantizeLinear_9<br>[1,12,77,2], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/ReduceSum_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_9 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_5 -.-> com_microsoft_DequantizeLinear_9
    Sigmoid_0["Sigmoid_0<br>[1,12,77,2], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Sigmoid_output_0"]
    classDef Sigmoid fill:#d3d3d3,stroke:#36454f,stroke-width:2px;;
    class Sigmoid_0 Sigmoid;
    com_microsoft_DequantizeLinear_9 -.-> Sigmoid_0
    com_microsoft_QuantizeLinear_6["com_microsoft_QuantizeLinear_6<br>[1,12,77,2], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/Sigmoid_output_0_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_6 com_microsoft_QuantizeLinear;
    Sigmoid_0 -.-> com_microsoft_QuantizeLinear_6
    com_microsoft_DequantizeLinear_10["com_microsoft_DequantizeLinear_10<br>[1,12,77,2], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear_Output/duplicated"]
    class com_microsoft_DequantizeLinear_10 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_6 -.-> com_microsoft_DequantizeLinear_10
    Slice_0["Slice_0<br>[1,12,77,1], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Slice_1_output_0"]
    com_microsoft_DequantizeLinear_10 -.-> Slice_0
    com_microsoft_QuantizeLinear_7["com_microsoft_QuantizeLinear_7<br>[1,12,77,1], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/Slice_1_output_0_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_7 com_microsoft_QuantizeLinear;
    Slice_0 -.-> com_microsoft_QuantizeLinear_7
    com_microsoft_DequantizeLinear_11["com_microsoft_DequantizeLinear_11<br>[1,12,77,1], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Slice_1_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_11 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_7 -.-> com_microsoft_DequantizeLinear_11
    com_microsoft_DequantizeLinear_12["com_microsoft_DequantizeLinear_12<br>[1,12,1,1], ty=1<br>lang_encoder.encoder.layer.0.attention.self.eco_a_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_12 com_microsoft_DequantizeLinear;
    Mul_0["Mul_0<br>[1,12,77,1], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Mul_2_output_0"]
    com_microsoft_DequantizeLinear_11 -.-> Mul_0
    com_microsoft_DequantizeLinear_12 -.-> Mul_0
    com_microsoft_QuantizeLinear_8["com_microsoft_QuantizeLinear_8<br>[1,12,77,1], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/Mul_2_output_0_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_8 com_microsoft_QuantizeLinear;
    Mul_0 -.-> com_microsoft_QuantizeLinear_8
    com_microsoft_DequantizeLinear_13["com_microsoft_DequantizeLinear_13<br>[1,12,77,1], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Mul_2_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_13 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_8 -.-> com_microsoft_DequantizeLinear_13
    Sub_0["Sub_0<br>[1,12,77,1], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Sub_output_0"]
    classDef Sub fill:#add8e6,stroke:#00008b,stroke-width:2px;;
    class Sub_0 Sub;
    com_microsoft_DequantizeLinear_13 -.-> Sub_0
    com_microsoft_DequantizeLinear_0 -.-> Sub_0
    com_microsoft_QuantizeLinear_9["com_microsoft_QuantizeLinear_9<br>[1,12,77,1], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/Sub_output_0_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_9 com_microsoft_QuantizeLinear;
    Sub_0 -.-> com_microsoft_QuantizeLinear_9
    com_microsoft_DequantizeLinear_14["com_microsoft_DequantizeLinear_14<br>[1,12,77,1], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Sub_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_14 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_9 -.-> com_microsoft_DequantizeLinear_14
    com_microsoft_DequantizeLinear_15["com_microsoft_DequantizeLinear_15<br>[1,12,77,2], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_15 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_6 -.-> com_microsoft_DequantizeLinear_15
    Slice_1["Slice_1<br>[1,12,77,1], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Slice_output_0"]
    com_microsoft_DequantizeLinear_15 -.-> Slice_1
    com_microsoft_QuantizeLinear_10["com_microsoft_QuantizeLinear_10<br>[1,12,77,1], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/Slice_output_0_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_10 com_microsoft_QuantizeLinear;
    Slice_1 -.-> com_microsoft_QuantizeLinear_10
    com_microsoft_DequantizeLinear_16["com_microsoft_DequantizeLinear_16<br>[1,12,77,1], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Slice_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_16 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_10 -.-> com_microsoft_DequantizeLinear_16
    Mul_1["Mul_1<br>[1,12,77,1], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Mul_3_output_0"]
    com_microsoft_DequantizeLinear_16 -.-> Mul_1
    com_microsoft_DequantizeLinear_14 -.-> Mul_1
    com_microsoft_QuantizeLinear_11["com_microsoft_QuantizeLinear_11<br>[1,12,77,1], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/Mul_3_output_0_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_11 com_microsoft_QuantizeLinear;
    Mul_1 -.-> com_microsoft_QuantizeLinear_11
    com_microsoft_DequantizeLinear_17["com_microsoft_DequantizeLinear_17<br>[1,12,77,1], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Mul_3_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_17 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_11 -.-> com_microsoft_DequantizeLinear_17
    com_microsoft_DequantizeLinear_18["com_microsoft_DequantizeLinear_18<br>[], ty=1<br>/lang_encoder/embeddings/LayerNorm/Constant_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_18 com_microsoft_DequantizeLinear;
    Add_1["Add_1<br>[1,12,77,1], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Add_2_output_0"]
    class Add_1 Add;
    com_microsoft_DequantizeLinear_17 -.-> Add_1
    com_microsoft_DequantizeLinear_18 -.-> Add_1
    com_microsoft_QuantizeLinear_12["com_microsoft_QuantizeLinear_12<br>[1,12,77,1], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/Add_2_output_0_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_12 com_microsoft_QuantizeLinear;
    Add_1 -.-> com_microsoft_QuantizeLinear_12
    com_microsoft_DequantizeLinear_19["com_microsoft_DequantizeLinear_19<br>[1,12,77,1], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Add_2_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_19 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_12 -.-> com_microsoft_DequantizeLinear_19
    com_microsoft_DequantizeLinear_20["com_microsoft_DequantizeLinear_20<br>[1,12,77,77], ty=1<br>/lang_encoder/GatherElements_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_20 com_microsoft_DequantizeLinear;
    Mul_2["Mul_2<br>[1,12,77,77], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Mul_4_output_0"]
    com_microsoft_DequantizeLinear_19 -.-> Mul_2
    com_microsoft_DequantizeLinear_20 -.-> Mul_2
    com_microsoft_QuantizeLinear_13["com_microsoft_QuantizeLinear_13<br>[1,12,77,77], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/Mul_4_output_0_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_13 com_microsoft_QuantizeLinear;
    Mul_2 -.-> com_microsoft_QuantizeLinear_13
    com_microsoft_DequantizeLinear_21["com_microsoft_DequantizeLinear_21<br>[1,12,77,77], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Mul_4_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_21 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_13 -.-> com_microsoft_DequantizeLinear_21
    com_microsoft_DequantizeLinear_22["com_microsoft_DequantizeLinear_22<br>[1,12,77,96], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_22 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_1 -.-> com_microsoft_DequantizeLinear_22
    input_1[\"input_1<br>[1,77,1152], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/key/MatMul_output_0_QuantizeLinear_Output"/]
    class input_1 input;
    com_microsoft_DequantizeLinear_23["com_microsoft_DequantizeLinear_23<br>[1,77,1152], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/key/MatMul_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_23 com_microsoft_DequantizeLinear;
    input_1 -.-> com_microsoft_DequantizeLinear_23
    Reshape_2["Reshape_2<br>[1,77,12,96], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Reshape_1_output_0"]
    class Reshape_2 Reshape;
    com_microsoft_DequantizeLinear_23 -.-> Reshape_2
    com_microsoft_QuantizeLinear_14["com_microsoft_QuantizeLinear_14<br>[1,77,12,96], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/Reshape_1_output_0_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_14 com_microsoft_QuantizeLinear;
    Reshape_2 -.-> com_microsoft_QuantizeLinear_14
    com_microsoft_DequantizeLinear_24["com_microsoft_DequantizeLinear_24<br>[1,77,12,96], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Reshape_1_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_24 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_14 -.-> com_microsoft_DequantizeLinear_24
    Transpose_1["Transpose_1<br>[1,12,96,77], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Transpose_2_output_0"]
    class Transpose_1 Transpose;
    com_microsoft_DequantizeLinear_24 -.-> Transpose_1
    com_microsoft_QuantizeLinear_15["com_microsoft_QuantizeLinear_15<br>[1,12,96,77], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/Transpose_2_output_0_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_15 com_microsoft_QuantizeLinear;
    Transpose_1 -.-> com_microsoft_QuantizeLinear_15
    com_microsoft_DequantizeLinear_25["com_microsoft_DequantizeLinear_25<br>[1,12,96,77], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Transpose_2_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_25 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_15 -.-> com_microsoft_DequantizeLinear_25
    MatMul_1["MatMul_1<br>[1,12,77,77], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/MatMul_output_0"]
    class MatMul_1 MatMul;
    com_microsoft_DequantizeLinear_22 -.-> MatMul_1
    com_microsoft_DequantizeLinear_25 -.-> MatMul_1
    com_microsoft_QuantizeLinear_16["com_microsoft_QuantizeLinear_16<br>[1,12,77,77], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/MatMul_output_0_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_16 com_microsoft_QuantizeLinear;
    MatMul_1 -.-> com_microsoft_QuantizeLinear_16
    com_microsoft_DequantizeLinear_26["com_microsoft_DequantizeLinear_26<br>[1,12,77,77], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/MatMul_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_26 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_16 -.-> com_microsoft_DequantizeLinear_26
    com_microsoft_DequantizeLinear_27["com_microsoft_DequantizeLinear_27<br>[], ty=1<br>/lang_encoder/Constant_17_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_27 com_microsoft_DequantizeLinear;
    Div_0["Div_0<br>[1,12,77,77], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Div_output_0"]
    com_microsoft_DequantizeLinear_26 -.-> Div_0
    com_microsoft_DequantizeLinear_27 -.-> Div_0
    com_microsoft_QuantizeLinear_17["com_microsoft_QuantizeLinear_17<br>[1,12,77,77], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/Div_output_0_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_17 com_microsoft_QuantizeLinear;
    Div_0 -.-> com_microsoft_QuantizeLinear_17
    com_microsoft_DequantizeLinear_28["com_microsoft_DequantizeLinear_28<br>[1,12,77,77], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Div_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_28 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_17 -.-> com_microsoft_DequantizeLinear_28
    input_2[\"input_2<br>[1,1,1,77], ty=4<br>/lang_encoder/Mul_output_0_QuantizeLinear_Output"/]
    class input_2 input;
    com_microsoft_DequantizeLinear_29["com_microsoft_DequantizeLinear_29<br>[1,1,1,77], ty=1<br>/lang_encoder/Mul_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_29 com_microsoft_DequantizeLinear;
    input_2 -.-> com_microsoft_DequantizeLinear_29
    Add_2["Add_2<br>[1,12,77,77], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Add_output_0"]
    class Add_2 Add;
    com_microsoft_DequantizeLinear_28 -.-> Add_2
    com_microsoft_DequantizeLinear_29 -.-> Add_2
    com_microsoft_QuantizeLinear_18["com_microsoft_QuantizeLinear_18<br>[1,12,77,77], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/Add_output_0_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_18 com_microsoft_QuantizeLinear;
    Add_2 -.-> com_microsoft_QuantizeLinear_18
    com_microsoft_DequantizeLinear_30["com_microsoft_DequantizeLinear_30<br>[1,12,77,77], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Add_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_30 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_18 -.-> com_microsoft_DequantizeLinear_30
    Add_3["Add_3<br>[1,12,77,77], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Add_3_output_0"]
    class Add_3 Add;
    com_microsoft_DequantizeLinear_30 -.-> Add_3
    com_microsoft_DequantizeLinear_21 -.-> Add_3
    com_microsoft_QuantizeLinear_19["com_microsoft_QuantizeLinear_19<br>[1,12,77,77], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/Add_3_output_0_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_19 com_microsoft_QuantizeLinear;
    Add_3 -.-> com_microsoft_QuantizeLinear_19
    com_microsoft_DequantizeLinear_31["com_microsoft_DequantizeLinear_31<br>[1,12,77,77], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Add_3_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_31 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_19 -.-> com_microsoft_DequantizeLinear_31
    Softmax_0["Softmax_0<br>[1,12,77,77], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Softmax_output_0"]
    com_microsoft_DequantizeLinear_31 -.-> Softmax_0
    com_microsoft_QuantizeLinear_20["com_microsoft_QuantizeLinear_20<br>[1,12,77,77], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/Softmax_output_0_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_20 com_microsoft_QuantizeLinear;
    Softmax_0 -.-> com_microsoft_QuantizeLinear_20
    com_microsoft_DequantizeLinear_32["com_microsoft_DequantizeLinear_32<br>[1,12,77,77], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Softmax_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_32 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_20 -.-> com_microsoft_DequantizeLinear_32
    input_3[\"input_3<br>[1,77,768], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/value/Add_output_0_QuantizeLinear_Output"/]
    class input_3 input;
    com_microsoft_DequantizeLinear_33["com_microsoft_DequantizeLinear_33<br>[1,77,768], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/value/Add_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_33 com_microsoft_DequantizeLinear;
    input_3 -.-> com_microsoft_DequantizeLinear_33
    Reshape_3["Reshape_3<br>[1,77,12,64], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Reshape_2_output_0"]
    class Reshape_3 Reshape;
    com_microsoft_DequantizeLinear_33 -.-> Reshape_3
    com_microsoft_QuantizeLinear_21["com_microsoft_QuantizeLinear_21<br>[1,77,12,64], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/Reshape_2_output_0_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_21 com_microsoft_QuantizeLinear;
    Reshape_3 -.-> com_microsoft_QuantizeLinear_21
    com_microsoft_DequantizeLinear_34["com_microsoft_DequantizeLinear_34<br>[1,77,12,64], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Reshape_2_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_34 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_21 -.-> com_microsoft_DequantizeLinear_34
    Transpose_2["Transpose_2<br>[1,12,77,64], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Transpose_1_output_0"]
    class Transpose_2 Transpose;
    com_microsoft_DequantizeLinear_34 -.-> Transpose_2
    com_microsoft_QuantizeLinear_22["com_microsoft_QuantizeLinear_22<br>[1,12,77,64], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/Transpose_1_output_0_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_22 com_microsoft_QuantizeLinear;
    Transpose_2 -.-> com_microsoft_QuantizeLinear_22
    com_microsoft_DequantizeLinear_35["com_microsoft_DequantizeLinear_35<br>[1,12,77,64], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Transpose_1_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_35 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_22 -.-> com_microsoft_DequantizeLinear_35
    MatMul_2["MatMul_2<br>[1,12,77,64], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/MatMul_1_output_0"]
    class MatMul_2 MatMul;
    com_microsoft_DequantizeLinear_32 -.-> MatMul_2
    com_microsoft_DequantizeLinear_35 -.-> MatMul_2
    com_microsoft_QuantizeLinear_23["com_microsoft_QuantizeLinear_23<br>[1,12,77,64], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/MatMul_1_output_0_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_23 com_microsoft_QuantizeLinear;
    MatMul_2 -.-> com_microsoft_QuantizeLinear_23
    com_microsoft_DequantizeLinear_36["com_microsoft_DequantizeLinear_36<br>[1,12,77,64], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/MatMul_1_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_36 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_23 -.-> com_microsoft_DequantizeLinear_36
    Transpose_3["Transpose_3<br>[1,77,12,64], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Transpose_3_output_0"]
    class Transpose_3 Transpose;
    com_microsoft_DequantizeLinear_36 -.-> Transpose_3
    com_microsoft_QuantizeLinear_24["com_microsoft_QuantizeLinear_24<br>[1,77,12,64], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/Transpose_3_output_0_QuantizeLinear_Output"]
    class com_microsoft_QuantizeLinear_24 com_microsoft_QuantizeLinear;
    Transpose_3 -.-> com_microsoft_QuantizeLinear_24
    com_microsoft_DequantizeLinear_37["com_microsoft_DequantizeLinear_37<br>[1,77,12,64], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Transpose_3_output_0_DequantizeLinear_Output"]
    class com_microsoft_DequantizeLinear_37 com_microsoft_DequantizeLinear;
    com_microsoft_QuantizeLinear_24 -.-> com_microsoft_DequantizeLinear_37
    Reshape_4["Reshape_4<br>[1,77,768], ty=1<br>/lang_encoder/encoder/layer.0/attention/self/Reshape_4_output_0"]
    class Reshape_4 Reshape;
    com_microsoft_DequantizeLinear_37 -.-> Reshape_4
    com_microsoft_QuantizeLinear_25[["com_microsoft_QuantizeLinear_25<br>[1,77,768], ty=4<br>/lang_encoder/encoder/layer.0/attention/self/Reshape_4_output_0_QuantizeLinear_Output"]]
    class com_microsoft_QuantizeLinear_25 com_microsoft_QuantizeLinear;
    Reshape_4 --o com_microsoft_QuantizeLinear_25
```

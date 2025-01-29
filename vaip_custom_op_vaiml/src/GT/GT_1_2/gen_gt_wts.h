/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once
#define gt_min(a, b) ((a < b) ? a : b)
#define gt_max(a, b) ((a < b) ? b : a)
#include "../common/gen_gt_wts_common.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdint.h>
#include <string.h>
#include <tuple>
#include <unordered_map>
#include <vector>
namespace vaip_vaiml_custom_op {
const std::map<std::string, std::map<std::string, float>> scale_maps = {
    {"GT_v1.2",
     {{"/linear_k/Add_output_0_scale", 8.286539377877489e-05f},
      {"/linear_k/MatMul_output_0_scale", 8.290805271826684e-05f},
      {"/linear_k/Transpose_output_0_scale", 2.867772309400607e-05f},
      {"/linear_q/Add_output_0_scale", 9.622959623811767e-05f},
      {"/linear_q/MatMul_output_0_scale", 7.274092058651149e-05f},
      {"/linear_q/Transpose_output_0_scale", 2.7388752641854808e-05f},
      {"/linear_v/Add_output_0_scale", 3.748925155377947e-05f},
      {"/linear_v/MatMul_output_0_scale", 3.835824099951424e-05f},
      {"/linear_v/Transpose_output_0_scale", 1.1351132343406789e-05f},
      {"/norm1/Add_1_output_0_scale", 1.5558491213596426e-05f},
      {"encoder.encoders.0.self_attn.linear_k.bias_scale",
       2.5396388991794083e-07f},
      {"encoder.encoders.0.self_attn.linear_q.bias_scale",
       6.11552968621254e-05f},
      {"encoder.encoders.0.self_attn.linear_v.bias_scale",
       3.932103936676867e-06f},
      {"/linear_k_1/Add_output_0_scale", 0.00012423511361703277f},
      {"/linear_k_1/MatMul_output_0_scale", 0.0001242503203684464f},
      {"/linear_k_1/Transpose_output_0_scale", 2.5362163796671666e-05f},
      {"/linear_q_1/Add_output_0_scale", 0.0001535257906652987f},
      {"/linear_q_1/MatMul_output_0_scale", 0.0001287784252781421f},
      {"/linear_q_1/Transpose_output_0_scale", 2.8558511985465884e-05f},
      {"/linear_v_1/Add_output_0_scale", 5.1995590183651075e-05f},
      {"/linear_v_1/MatMul_output_0_scale", 5.181293090572581e-05f},
      {"/linear_v_1/Transpose_output_0_scale", 1.1556026947801001e-05f},
      {"/norm1_1/Add_1_output_0_scale", 2.5268469471484423e-05f},
      {"encoder.encoders.1.self_attn.linear_k.bias_scale",
       3.4447819530214474e-07f},
      {"encoder.encoders.1.self_attn.linear_q.bias_scale",
       3.1471732654608786e-05f},
      {"encoder.encoders.1.self_attn.linear_v.bias_scale",
       5.184751898923423e-06f},
      {"/linear_k_2/Add_output_0_scale", 0.0001397366722812876f},
      {"/linear_k_2/MatMul_output_0_scale", 0.00013980804942548275f},
      {"/linear_k_2/Transpose_output_0_scale", 2.265011062263511e-05f},
      {"/linear_q_2/Add_output_0_scale", 0.00021069352806080133f},
      {"/linear_q_2/MatMul_output_0_scale", 0.00019570380391087383f},
      {"/linear_q_2/Transpose_output_0_scale", 3.512257171678357e-05f},
      {"/linear_v_2/Add_output_0_scale", 7.350883970502764e-05f},
      {"/linear_v_2/MatMul_output_0_scale", 7.352211105171591e-05f},
      {"/linear_v_2/Transpose_output_0_scale", 1.1868955880345311e-05f},
      {"/norm1_2/Add_1_output_0_scale", 4.811190228792839e-05f},
      {"encoder.encoders.2.self_attn.linear_k.bias_scale",
       2.853383023193601e-07f},
      {"encoder.encoders.2.self_attn.linear_q.bias_scale",
       3.64636980521027e-05f},
      {"encoder.encoders.2.self_attn.linear_v.bias_scale",
       1.322498292211094e-06f},
      {"/linear_k_3/Add_output_0_scale", 9.464300092076883e-05f},
      {"/linear_k_3/MatMul_output_0_scale", 9.4661554612685e-05f},
      {"/linear_k_3/Transpose_output_0_scale", 1.8276552509632893e-05f},
      {"/linear_q_3/Add_output_0_scale", 0.0001419164618710056f},
      {"/linear_q_3/MatMul_output_0_scale", 0.0001114356709877029f},
      {"/linear_q_3/Transpose_output_0_scale", 3.112900230917148e-05f},
      {"/linear_v_3/Add_output_0_scale", 5.5827360483817756e-05f},
      {"/linear_v_3/MatMul_output_0_scale", 5.609372237813659e-05f},
      {"/linear_v_3/Transpose_output_0_scale", 1.2871074432041496e-05f},
      {"/norm1_3/Add_1_output_0_scale", 5.8220433857059106e-05f},
      {"encoder.encoders.3.self_attn.linear_k.bias_scale",
       5.501872237800853e-07f},
      {"encoder.encoders.3.self_attn.linear_q.bias_scale",
       3.048079088330269e-05f},
      {"encoder.encoders.3.self_attn.linear_v.bias_scale",
       2.2072681531426497e-06f},
      {"/linear_k_4/Add_output_0_scale", 0.00012817303650081158f},
      {"/linear_k_4/MatMul_output_0_scale", 0.00012821979180444032f},
      {"/linear_k_4/Transpose_output_0_scale", 2.5585684852558188e-05f},
      {"/linear_q_4/Add_output_0_scale", 0.00011153003288200125f},
      {"/linear_q_4/MatMul_output_0_scale", 0.00011396736226743087f},
      {"/linear_q_4/Transpose_output_0_scale", 2.1018408006057143e-05f},
      {"/linear_v_4/Add_output_0_scale", 9.074358240468428e-05f},
      {"/linear_v_4/MatMul_output_0_scale", 9.24875057535246e-05f},
      {"/linear_v_4/Transpose_output_0_scale", 1.4242002180253621e-05f},
      {"/norm1_4/Add_1_output_0_scale", 5.0162871048087254e-05f},
      {"encoder.encoders.4.self_attn.linear_k.bias_scale",
       4.0420016489406407e-07f},
      {"encoder.encoders.4.self_attn.linear_q.bias_scale",
       2.7157781005371362e-05f},
      {"encoder.encoders.4.self_attn.linear_v.bias_scale",
       2.753962917267927e-06f},
      {"/linear_k_5/Add_output_0_scale", 9.326435247203335e-05f},
      {"/linear_k_5/MatMul_output_0_scale", 9.332823537988588e-05f},
      {"/linear_k_5/Transpose_output_0_scale", 2.110036621161271e-05f},
      {"/linear_q_5/Add_output_0_scale", 0.00016566617705393583f},
      {"/linear_q_5/MatMul_output_0_scale", 0.00015716120833531022f},
      {"/linear_q_5/Transpose_output_0_scale", 1.914828499138821e-05f},
      {"/linear_v_5/Add_output_0_scale", 8.376888581551611e-05f},
      {"/linear_v_5/MatMul_output_0_scale", 8.734522270970047e-05f},
      {"/linear_v_5/Transpose_output_0_scale", 1.3176552783988882e-05f},
      {"/norm1_5/Add_1_output_0_scale", 8.205202902900055e-05f},
      {"encoder.encoders.5.self_attn.linear_k.bias_scale",
       9.867512744676787e-07f},
      {"encoder.encoders.5.self_attn.linear_q.bias_scale",
       2.1420746634248644e-05f},
      {"encoder.encoders.5.self_attn.linear_v.bias_scale",
       3.5847153867507586e-06f},
      {"/linear_k_6/Add_output_0_scale", 9.256654448108748e-05f},
      {"/linear_k_6/MatMul_output_0_scale", 9.255334589397535e-05f},
      {"/linear_k_6/Transpose_output_0_scale", 2.17783799598692e-05f},
      {"/linear_q_6/Add_output_0_scale", 0.00010673625365598127f},
      {"/linear_q_6/MatMul_output_0_scale", 0.0001102297319448553f},
      {"/linear_q_6/Transpose_output_0_scale", 1.7173850210383534e-05f},
      {"/linear_v_6/Add_output_0_scale", 6.962427141843364e-05f},
      {"/linear_v_6/MatMul_output_0_scale", 7.060423376969993e-05f},
      {"/linear_v_6/Transpose_output_0_scale", 1.3463404684443958e-05f},
      {"/norm1_6/Add_1_output_0_scale", 7.652201020391658e-05f},
      {"encoder.encoders.6.self_attn.linear_k.bias_scale",
       3.7614364600813133e-07f},
      {"encoder.encoders.6.self_attn.linear_q.bias_scale",
       2.0638422938645817e-05f},
      {"encoder.encoders.6.self_attn.linear_v.bias_scale",
       2.9691016152355587e-06f},
      {"/linear_k_7/Add_output_0_scale", 0.00010678498802008107f},
      {"/linear_k_7/MatMul_output_0_scale", 0.00010678498802008107f},
      {"/linear_k_7/Transpose_output_0_scale", 2.042235246335622e-05f},
      {"/linear_q_7/Add_output_0_scale", 0.00013290146307554096f},
      {"/linear_q_7/MatMul_output_0_scale", 0.00011111564526800066f},
      {"/linear_q_7/Transpose_output_0_scale", 1.9431410692050122e-05f},
      {"/linear_v_7/Add_output_0_scale", 9.735547791933641e-05f},
      {"/linear_v_7/MatMul_output_0_scale", 9.735547791933641e-05f},
      {"/linear_v_7/Transpose_output_0_scale", 1.4916289728716947e-05f},
      {"/norm1_7/Add_1_output_0_scale", 9.283765393774956e-05f},
      {"encoder.encoders.7.self_attn.linear_k.bias_scale",
       8.37038953704905e-07f},
      {"encoder.encoders.7.self_attn.linear_q.bias_scale",
       2.49076711043017e-05f},
      {"encoder.encoders.7.self_attn.linear_v.bias_scale",
       2.2873632588016335e-06f},
      {"/linear_k_8/Add_output_0_scale", 0.00013839152234140784f},
      {"/linear_k_8/MatMul_output_0_scale", 0.00013823674817103893f},
      {"/linear_k_8/Transpose_output_0_scale", 1.707699084363412e-05f},
      {"/linear_q_8/Add_output_0_scale", 0.0005069079925306141f},
      {"/linear_q_8/MatMul_output_0_scale", 0.0005070346524007618f},
      {"/linear_q_8/Transpose_output_0_scale", 3.5182179999537766e-05f},
      {"/linear_v_8/Add_output_0_scale", 0.0001183806307381019f},
      {"/linear_v_8/MatMul_output_0_scale", 0.00011865478882100433f},
      {"/linear_v_8/Transpose_output_0_scale", 1.646975943003781e-05f},
      {"/norm1_8/Add_1_output_0_scale", 0.00010080249194288626f},
      {"encoder.encoders.8.self_attn.linear_k.bias_scale",
       1.3904858633395634e-06f},
      {"encoder.encoders.8.self_attn.linear_q.bias_scale",
       3.379635018063709e-05f},
      {"encoder.encoders.8.self_attn.linear_v.bias_scale",
       4.712564077635761e-06f},
      {"/linear_k_9/Add_output_0_scale", 0.00015599267499055713f},
      {"/linear_k_9/MatMul_output_0_scale", 0.00015589695249218494f},
      {"/linear_k_9/Transpose_output_0_scale", 2.058626887446735e-05f},
      {"/linear_q_9/Add_output_0_scale", 0.00023486198915634304f},
      {"/linear_q_9/MatMul_output_0_scale", 0.0002233053237432614f},
      {"/linear_q_9/Transpose_output_0_scale", 2.265011062263511e-05f},
      {"/linear_v_9/Add_output_0_scale", 0.00011769228876801208f},
      {"/linear_v_9/MatMul_output_0_scale", 0.00011720543261617422f},
      {"/linear_v_9/Transpose_output_0_scale", 2.008707269851584e-05f},
      {"/norm1_9/Add_1_output_0_scale", 8.996299584396183e-05f},
      {"encoder.encoders.9.self_attn.linear_k.bias_scale",
       4.3586561560005066e-07f},
      {"encoder.encoders.9.self_attn.linear_q.bias_scale",
       2.54962760664057e-05f},
      {"encoder.encoders.9.self_attn.linear_v.bias_scale",
       5.565668743656715e-06f},
      {"/linear_k_10/Add_output_0_scale", 0.00010930885764537379f},
      {"/linear_k_10/MatMul_output_0_scale", 0.00010925201786449179f},
      {"/linear_k_10/Transpose_output_0_scale", 1.6212710761465132e-05f},
      {"/linear_q_10/Add_output_0_scale", 0.00017946540901903063f},
      {"/linear_q_10/MatMul_output_0_scale", 0.00017479941016063094f},
      {"/linear_q_10/Transpose_output_0_scale", 2.2486196030513383e-05f},
      {"/linear_v_10/Add_output_0_scale", 8.202694880310446e-05f},
      {"/linear_v_10/MatMul_output_0_scale", 8.067471935646608e-05f},
      {"/linear_v_10/Transpose_output_0_scale", 1.2368152965791523e-05f},
      {"/norm1_10/Add_1_output_0_scale", 8.713566057849675e-05f},
      {"encoder.encoders.10.self_attn.linear_k.bias_scale",
       6.659058158220432e-07f},
      {"encoder.encoders.10.self_attn.linear_q.bias_scale",
       2.399123513896484e-05f},
      {"encoder.encoders.10.self_attn.linear_v.bias_scale",
       5.623411652777577e-06f},
      {"/linear_k_11/Add_output_0_scale", 0.00012250812142156065f},
      {"/linear_k_11/MatMul_output_0_scale", 0.00012242757657077163f},
      {"/linear_k_11/Transpose_output_0_scale", 1.4618261957366485e-05f},
      {"/linear_q_11/Add_output_0_scale", 0.00020044553093612194f},
      {"/linear_q_11/MatMul_output_0_scale", 0.00019360618898645043f},
      {"/linear_q_11/Transpose_output_0_scale", 2.5988021661760285e-05f},
      {"/linear_v_11/Add_output_0_scale", 0.00011248695227550343f},
      {"/linear_v_11/MatMul_output_0_scale", 0.00011071275366703048f},
      {"/linear_v_11/Transpose_output_0_scale", 1.2338349733909126e-05f},
      {"/norm1_11/Add_1_output_0_scale", 0.0001049437269102782f},
      {"encoder.encoders.11.self_attn.linear_k.bias_scale",
       7.695170438637433e-07f},
      {"encoder.encoders.11.self_attn.linear_q.bias_scale",
       2.7992258765152656e-05f},
      {"encoder.encoders.11.self_attn.linear_v.bias_scale",
       4.684623945649946e-06f},
      {"/linear_k_12/Add_output_0_scale", 0.00014686973008792847f},
      {"/linear_k_12/MatMul_output_0_scale", 0.0001468479895265773f},
      {"/linear_k_12/Transpose_output_0_scale", 2.162191412935499e-05f},
      {"/linear_q_12/Add_output_0_scale", 0.00030412987689487636f},
      {"/linear_q_12/MatMul_output_0_scale", 0.0003065606579184532f},
      {"/linear_q_12/Transpose_output_0_scale", 2.0221184968249872e-05f},
      {"/linear_v_12/Add_output_0_scale", 0.00011360452481312677f},
      {"/linear_v_12/MatMul_output_0_scale", 0.00011307075328659266f},
      {"/linear_v_12/Transpose_output_0_scale", 1.3619869605463464e-05f},
      {"/norm1_12/Add_1_output_0_scale", 9.755034261615947e-05f},
      {"encoder.encoders.12.self_attn.linear_k.bias_scale",
       6.96407084888051e-07f},
      {"encoder.encoders.12.self_attn.linear_q.bias_scale",
       2.64574155153241e-05f},
      {"encoder.encoders.12.self_attn.linear_v.bias_scale",
       2.98027771350462e-06f},
      {"/linear_k_13/Add_output_0_scale", 0.00020529770699795336f},
      {"/linear_k_13/MatMul_output_0_scale", 0.00020590050553437322f},
      {"/linear_k_13/Transpose_output_0_scale", 1.8134989659301937e-05f},
      {"/linear_q_13/Add_output_0_scale", 0.00022153621830511838f},
      {"/linear_q_13/MatMul_output_0_scale", 0.00019817083375528455f},
      {"/linear_q_13/Transpose_output_0_scale", 1.8112637917511165e-05f},
      {"/linear_v_13/Add_output_0_scale", 0.0001264085731236264f},
      {"/linear_v_13/MatMul_output_0_scale", 0.00012276750931050628f},
      {"/linear_v_13/Transpose_output_0_scale", 1.6607596990070306e-05f},
      {"/norm1_13/Add_1_output_0_scale", 0.00010728801862569526f},
      {"encoder.encoders.13.self_attn.linear_k.bias_scale",
       1.9762967440328794e-06f},
      {"encoder.encoders.13.self_attn.linear_q.bias_scale",
       2.569744538050145e-05f},
      {"encoder.encoders.13.self_attn.linear_v.bias_scale",
       5.396165306592593e-06f},
      {"/linear_k_14/Add_output_0_scale", 0.00013713071530219167f},
      {"/linear_k_14/MatMul_output_0_scale", 0.00013711345673073083f},
      {"/linear_k_14/Transpose_output_0_scale", 1.466296635044273e-05f},
      {"/linear_q_14/Add_output_0_scale", 0.0001850676053436473f},
      {"/linear_q_14/MatMul_output_0_scale", 0.00017009436851367354f},
      {"/linear_q_14/Transpose_output_0_scale", 2.02770643227268e-05f},
      {"/linear_v_14/Add_output_0_scale", 0.00011202471796423197f},
      {"/linear_v_14/MatMul_output_0_scale", 0.00011178486602148041f},
      {"/linear_v_14/Transpose_output_0_scale", 1.2744412742904387e-05f},
      {"/norm1_14/Add_1_output_0_scale", 0.00010596121865091845f},
      {"encoder.encoders.14.self_attn.linear_k.bias_scale",
       5.008263315176009e-07f},
      {"encoder.encoders.14.self_attn.linear_q.bias_scale",
       2.3119504476198927e-05f},
      {"encoder.encoders.14.self_attn.linear_v.bias_scale",
       3.444083404247067e-06f},
      {"/linear_k_15/Add_output_0_scale", 0.00014357810141518712f},
      {"/linear_k_15/MatMul_output_0_scale", 0.00014330766862258315f},
      {"/linear_k_15/Transpose_output_0_scale", 1.5143536074901931e-05f},
      {"/linear_q_15/Add_output_0_scale", 0.00015955735580064356f},
      {"/linear_q_15/MatMul_output_0_scale", 0.00013894661969970912f},
      {"/linear_q_15/Transpose_output_0_scale", 1.7188751371577382e-05f},
      {"/linear_v_15/Add_output_0_scale", 9.764233254827559e-05f},
      {"/linear_v_15/MatMul_output_0_scale", 9.740450332174078e-05f},
      {"/linear_v_15/Transpose_output_0_scale", 1.2371878256089985e-05f},
      {"/norm1_15/Add_1_output_0_scale", 9.83487771009095e-05f},
      {"encoder.encoders.15.self_attn.linear_k.bias_scale",
       7.585738330817549e-07f},
      {"encoder.encoders.15.self_attn.linear_q.bias_scale",
       2.533236147428397e-05f},
      {"encoder.encoders.15.self_attn.linear_v.bias_scale",
       2.8154311166872503e-06f},
      {"/linear_k_16/Add_output_0_scale", 0.00015407621685881168f},
      {"/linear_k_16/MatMul_output_0_scale", 0.0001542404934298247f},
      {"/linear_k_16/Transpose_output_0_scale", 1.5996640286175534e-05f},
      {"/linear_q_16/Add_output_0_scale", 0.0001720675063552335f},
      {"/linear_q_16/MatMul_output_0_scale", 0.00014396234473679215f},
      {"/linear_q_16/Transpose_output_0_scale", 1.966983290913049e-05f},
      {"/linear_v_16/Add_output_0_scale", 0.00010277327965013683f},
      {"/linear_v_16/MatMul_output_0_scale", 0.00010243367432849482f},
      {"/linear_v_16/Transpose_output_0_scale", 1.153367520601023e-05f},
      {"/norm1_16/Add_1_output_0_scale", 0.00011939035903196782f},
      {"encoder.encoders.16.self_attn.linear_k.bias_scale",
       1.5045745840325253e-06f},
      {"encoder.encoders.16.self_attn.linear_q.bias_scale",
       3.2865013054106385e-05f},
      {"encoder.encoders.16.self_attn.linear_v.bias_scale",
       3.5903033221984515e-06f},
      {"/linear_k_17/Add_output_0_scale", 0.0001593720371602103f},
      {"/linear_k_17/MatMul_output_0_scale", 0.00015948658983688802f},
      {"/linear_k_17/Transpose_output_0_scale", 1.4849233593849931e-05f},
      {"/linear_q_17/Add_output_0_scale", 0.0001730574731482193f},
      {"/linear_q_17/MatMul_output_0_scale", 0.00015699081995990127f},
      {"/linear_q_17/Transpose_output_0_scale", 2.0481958927121013e-05f},
      {"/linear_v_17/Add_output_0_scale", 0.00010063254740089178f},
      {"/linear_v_17/MatMul_output_0_scale", 0.00010095199104398489f},
      {"/linear_v_17/Transpose_output_0_scale", 1.0747626220108941e-05f},
      {"/norm1_17/Add_1_output_0_scale", 0.00011001220264006406f},
      {"encoder.encoders.17.self_attn.linear_k.bias_scale",
       2.357213361392496e-06f},
      {"encoder.encoders.17.self_attn.linear_q.bias_scale",
       2.8767130061169155e-05f},
      {"encoder.encoders.17.self_attn.linear_v.bias_scale",
       2.8787619612558046e-06f},
      {"/linear_k_18/Add_output_0_scale", 0.00016141659580171108f},
      {"/linear_k_18/MatMul_output_0_scale", 0.0001615083310753107f},
      {"/linear_k_18/Transpose_output_0_scale", 1.5735866327304393e-05f},
      {"/linear_q_18/Add_output_0_scale", 0.00017456304340157658f},
      {"/linear_q_18/MatMul_output_0_scale", 0.0001532950409455225f},
      {"/linear_q_18/Transpose_output_0_scale", 1.7501681213616394e-05f},
      {"/linear_v_18/Add_output_0_scale", 0.00011140156857436523f},
      {"/linear_v_18/MatMul_output_0_scale", 0.00011106191959697753f},
      {"/linear_v_18/Transpose_output_0_scale", 1.0442147868161555e-05f},
      {"/norm1_18/Add_1_output_0_scale", 0.00010688421025406569f},
      {"encoder.encoders.18.self_attn.linear_k.bias_scale",
       8.174808385774668e-07f},
      {"encoder.encoders.18.self_attn.linear_q.bias_scale",
       2.6099782189703546e-05f},
      {"encoder.encoders.18.self_attn.linear_v.bias_scale",
       1.9283327219454804e-06f},
      {"/linear_k_19/Add_output_0_scale", 0.00020017744100186974f},
      {"/linear_k_19/MatMul_output_0_scale", 0.00020346151723060757f},
      {"/linear_k_19/Transpose_output_0_scale", 1.5329804227803834e-05f},
      {"/linear_q_19/Add_output_0_scale", 0.000278785708360374f},
      {"/linear_q_19/MatMul_output_0_scale", 0.0002611424424685538f},
      {"/linear_q_19/Transpose_output_0_scale", 2.2232872652239166e-05f},
      {"/linear_v_19/Add_output_0_scale", 0.00016050689737312496f},
      {"/linear_v_19/MatMul_output_0_scale", 0.00015768541197758168f},
      {"/linear_v_19/Transpose_output_0_scale", 1.037509173329454e-05f},
      {"/norm1_19/Add_1_output_0_scale", 0.00011019562953151762f},
      {"encoder.encoders.19.self_attn.linear_k.bias_scale",
       5.9344779401726555e-06f},
      {"encoder.encoders.19.self_attn.linear_q.bias_scale",
       2.3872024030424654e-05f},
      {"encoder.encoders.19.self_attn.linear_v.bias_scale",
       3.7830900510016363e-06f},
      {"/linear_k_20/Add_output_0_scale", 0.00014745414955541492f},
      {"/linear_k_20/MatMul_output_0_scale", 0.00014793543959967792f},
      {"/linear_k_20/Transpose_output_0_scale", 1.4145142813504208e-05f},
      {"/linear_q_20/Add_output_0_scale", 0.00017778229084797204f},
      {"/linear_q_20/MatMul_output_0_scale", 0.0001554093323647976f},
      {"/linear_q_20/Transpose_output_0_scale", 1.8611834093462676e-05f},
      {"/linear_v_20/Add_output_0_scale", 0.00011425800039432943f},
      {"/linear_v_20/MatMul_output_0_scale", 0.00011237844591960311f},
      {"/linear_v_20/Transpose_output_0_scale", 1.0550183105806354e-05f},
      {"/norm1_20/Add_1_output_0_scale", 0.00010712543735280633f},
      {"encoder.encoders.20.self_attn.linear_k.bias_scale",
       1.448228658773587e-06f},
      {"encoder.encoders.20.self_attn.linear_q.bias_scale",
       2.493002284609247e-05f},
      {"encoder.encoders.20.self_attn.linear_v.bias_scale",
       2.755825562417158e-06f},
      {"/linear_k_21/Add_output_0_scale", 0.00018875901878345758f},
      {"/linear_k_21/MatMul_output_0_scale", 0.00018858094699680805f},
      {"/linear_k_21/Transpose_output_0_scale", 1.6149379007401876e-05f},
      {"/linear_q_21/Add_output_0_scale", 0.0001975385966943577f},
      {"/linear_q_21/MatMul_output_0_scale", 0.0001781680475687608f},
      {"/linear_q_21/Transpose_output_0_scale", 2.0742732885992154e-05f},
      {"/linear_v_21/Add_output_0_scale", 0.000132214481709525f},
      {"/linear_v_21/MatMul_output_0_scale", 0.00013067893451079726f},
      {"/linear_v_21/Transpose_output_0_scale", 1.0509204003028572e-05f},
      {"/norm1_21/Add_1_output_0_scale", 0.0001089268407667987f},
      {"encoder.encoders.21.self_attn.linear_k.bias_scale",
       1.7700056105240947e-06f},
      {"encoder.encoders.21.self_attn.linear_q.bias_scale",
       2.316420795978047e-05f},
      {"encoder.encoders.21.self_attn.linear_v.bias_scale",
       2.017741053350619e-06f},
      {"/linear_k_22/Add_output_0_scale", 0.0001830575056374073f},
      {"/linear_k_22/MatMul_output_0_scale", 0.00018362562695983797f},
      {"/linear_k_22/Transpose_output_0_scale", 1.5110008462215774e-05f},
      {"/linear_q_22/Add_output_0_scale", 0.0002111662906827405f},
      {"/linear_q_22/MatMul_output_0_scale", 0.00019435379363130778f},
      {"/linear_q_22/Transpose_output_0_scale", 2.0116875020903535e-05f},
      {"/linear_v_22/Add_output_0_scale", 0.00012138996680732816f},
      {"/linear_v_22/MatMul_output_0_scale", 0.00012132077245041728f},
      {"/linear_v_22/Transpose_output_0_scale", 1.0717823897721246e-05f},
      {"/norm1_22/Add_1_output_0_scale", 0.0001137102663051337f},
      {"encoder.encoders.22.self_attn.linear_k.bias_scale",
       2.6133309347642353e-06f},
      {"encoder.encoders.22.self_attn.linear_q.bias_scale",
       2.415515155007597e-05f},
      {"encoder.encoders.22.self_attn.linear_v.bias_scale",
       2.340449327675742e-06f},
      {"/linear_k_23/Add_output_0_scale", 0.00017657950229477137f},
      {"/linear_k_23/MatMul_output_0_scale", 0.00017576621030457318f},
      {"/linear_k_23/Transpose_output_0_scale", 1.5478817658731714e-05f},
      {"/linear_q_23/Add_output_0_scale", 0.0001949831930687651f},
      {"/linear_q_23/MatMul_output_0_scale", 0.00017417340131942183f},
      {"/linear_q_23/Transpose_output_0_scale", 1.8351060134591535e-05f},
      {"/linear_v_23/Add_output_0_scale", 0.00011273749259999022f},
      {"/linear_v_23/MatMul_output_0_scale", 0.00011213934340048581f},
      {"/linear_v_23/Transpose_output_0_scale", 1.096369669539854e-05f},
      {"/norm1_23/Add_1_output_0_scale", 0.00011844761320389807f},
      {"encoder.encoders.23.self_attn.linear_k.bias_scale",
       1.0961833822875633e-06f},
      {"encoder.encoders.23.self_attn.linear_q.bias_scale",
       2.1830534024047665e-05f},
      {"encoder.encoders.23.self_attn.linear_v.bias_scale",
       1.7914262571139261e-06f},
      {"/linear_k_24/Add_output_0_scale", 0.00023609769414179027f},
      {"/linear_k_24/MatMul_output_0_scale", 0.000236031279200688f},
      {"/linear_k_24/Transpose_output_0_scale", 1.760599116096273e-05f},
      {"/linear_q_24/Add_output_0_scale", 0.00024262841907329857f},
      {"/linear_q_24/MatMul_output_0_scale", 0.00021985349303577095f},
      {"/linear_q_24/Transpose_output_0_scale", 2.027333903242834e-05f},
      {"/linear_v_24/Add_output_0_scale", 0.0001317093410762027f},
      {"/linear_v_24/MatMul_output_0_scale", 0.00013117371418047696f},
      {"/linear_v_24/Transpose_output_0_scale", 1.074390092981048e-05f},
      {"/norm1_24/Add_1_output_0_scale", 0.00012072630488546565f},
      {"encoder.encoders.24.self_attn.linear_k.bias_scale",
       2.314371840839158e-06f},
      {"encoder.encoders.24.self_attn.linear_q.bias_scale",
       2.4885319362510927e-05f},
      {"encoder.encoders.24.self_attn.linear_v.bias_scale",
       2.4456903702230193e-06f},
      {"/linear_k_25/Add_output_0_scale", 0.00023472582688555121f},
      {"/linear_k_25/MatMul_output_0_scale", 0.0002346937544643879f},
      {"/linear_k_25/Transpose_output_0_scale", 1.7218553693965077e-05f},
      {"/linear_q_25/Add_output_0_scale", 0.00021872231445740908f},
      {"/linear_q_25/MatMul_output_0_scale", 0.0002040348044829443f},
      {"/linear_q_25/Transpose_output_0_scale", 1.453257937100716e-05f},
      {"/linear_v_25/Add_output_0_scale", 0.00012161253835074604f},
      {"/linear_v_25/MatMul_output_0_scale", 0.0001215937954839319f},
      {"/linear_v_25/Transpose_output_0_scale", 1.063586569216568e-05f},
      {"/norm1_25/Add_1_output_0_scale", 0.00010820248280651867f},
      {"encoder.encoders.25.self_attn.linear_k.bias_scale",
       1.4524197240461945e-06f},
      {"encoder.encoders.25.self_attn.linear_q.bias_scale",
       1.929729842231609e-05f},
      {"encoder.encoders.25.self_attn.linear_v.bias_scale",
       1.6596421801295946e-06f},
      {"/linear_k_26/Add_output_0_scale", 0.000219619381823577f},
      {"/linear_k_26/MatMul_output_0_scale", 0.00021992700931150466f},
      {"/linear_k_26/Transpose_output_0_scale", 1.712914672680199e-05f},
      {"/linear_q_26/Add_output_0_scale", 0.00025464888312853873f},
      {"/linear_q_26/MatMul_output_0_scale", 0.00023345164663624018f},
      {"/linear_q_26/Transpose_output_0_scale", 1.6600146409473382e-05f},
      {"/linear_v_26/Add_output_0_scale", 0.00012710431474260986f},
      {"/linear_v_26/MatMul_output_0_scale", 0.0001267540210392326f},
      {"/linear_v_26/Transpose_output_0_scale", 1.1451717000454664e-05f},
      {"/norm1_26/Add_1_output_0_scale", 0.00013120197399985045f},
      {"encoder.encoders.26.self_attn.linear_k.bias_scale",
       1.0994431249855552e-06f},
      {"encoder.encoders.26.self_attn.linear_q.bias_scale",
       2.2933236323297024e-05f},
      {"encoder.encoders.26.self_attn.linear_v.bias_scale",
       2.397260914221988e-06f},
      {"/linear_k_27/Add_output_0_scale", 0.00016913405852392316f},
      {"/linear_k_27/MatMul_output_0_scale", 0.00016901975322980434f},
      {"/linear_k_27/Transpose_output_0_scale", 1.646603413973935e-05f},
      {"/linear_q_27/Add_output_0_scale", 0.00020405923714861274f},
      {"/linear_q_27/MatMul_output_0_scale", 0.00018476939294487238f},
      {"/linear_q_27/Transpose_output_0_scale", 1.72558084159391e-05f},
      {"/linear_v_27/Add_output_0_scale", 0.00012739509111270308f},
      {"/linear_v_27/MatMul_output_0_scale", 0.0001274492242373526f},
      {"/linear_v_27/Transpose_output_0_scale", 1.2044047252857126e-05f},
      {"/norm1_27/Add_1_output_0_scale", 0.00013740669237449765f},
      {"encoder.encoders.27.self_attn.linear_k.bias_scale",
       1.6242513538600178e-06f},
      {"encoder.encoders.27.self_attn.linear_q.bias_scale",
       2.2694814106216654e-05f},
      {"encoder.encoders.27.self_attn.linear_v.bias_scale",
       3.502757635942544e-06f},
      {"/linear_k_28/Add_output_0_scale", 0.00015404332953039557f},
      {"/linear_k_28/MatMul_output_0_scale", 0.00015391054330393672f},
      {"/linear_k_28/Transpose_output_0_scale", 1.5594303476973437e-05f},
      {"/linear_q_28/Add_output_0_scale", 0.00019147850980516523f},
      {"/linear_q_28/MatMul_output_0_scale", 0.00017306861991528422f},
      {"/linear_q_28/Transpose_output_0_scale", 1.5504894690820947e-05f},
      {"/linear_v_28/Add_output_0_scale", 0.0001226232125191018f},
      {"/linear_v_28/MatMul_output_0_scale", 0.00012359315587673336f},
      {"/linear_v_28/Transpose_output_0_scale", 1.1719942449417431e-05f},
      {"/norm1_28/Add_1_output_0_scale", 0.00014996349636930972f},
      {"encoder.encoders.28.self_attn.linear_k.bias_scale",
       3.4487400171201443e-06f},
      {"encoder.encoders.28.self_attn.linear_q.bias_scale",
       2.0250987290637568e-05f},
      {"encoder.encoders.28.self_attn.linear_v.bias_scale",
       2.5397553145012353e-06f},
      {"/linear_k_29/Add_output_0_scale", 0.0001701028668321669f},
      {"/linear_k_29/MatMul_output_0_scale", 0.00017015707271639258f},
      {"/linear_k_29/Transpose_output_0_scale", 1.598546441528015e-05f},
      {"/linear_q_29/Add_output_0_scale", 0.00022759442799724638f},
      {"/linear_q_29/MatMul_output_0_scale", 0.00021303204994183034f},
      {"/linear_q_29/Transpose_output_0_scale", 1.531117595732212e-05f},
      {"/linear_v_29/Add_output_0_scale", 0.00013392514665611088f},
      {"/linear_v_29/MatMul_output_0_scale", 0.00013418176968116313f},
      {"/linear_v_29/Transpose_output_0_scale", 1.1760920642700512e-05f},
      {"/norm1_29/Add_1_output_0_scale", 0.00015352069749496877f},
      {"encoder.encoders.29.self_attn.linear_k.bias_scale",
       1.2936268376506632e-06f},
      {"encoder.encoders.29.self_attn.linear_q.bias_scale",
       2.0489409507717937e-05f},
      {"encoder.encoders.29.self_attn.linear_v.bias_scale",
       4.934222488373052e-06f},
      {"/linear_k_30/Add_output_0_scale", 0.00018713987083174288f},
      {"/linear_k_30/MatMul_output_0_scale", 0.0001889843842945993f},
      {"/linear_k_30/Transpose_output_0_scale", 1.530745066702366e-05f},
      {"/linear_q_30/Add_output_0_scale", 0.0004271575016900897f},
      {"/linear_q_30/MatMul_output_0_scale", 0.0004271575016900897f},
      {"/linear_q_30/Transpose_output_0_scale", 1.746070120134391e-05f},
      {"/linear_v_30/Add_output_0_scale", 0.00013382572797127068f},
      {"/linear_v_30/MatMul_output_0_scale", 0.0001338486617896706f},
      {"/linear_v_30/Transpose_output_0_scale", 1.1574653399293311e-05f},
      {"/norm1_30/Add_1_output_0_scale", 0.00017387702246196568f},
      {"encoder.encoders.30.self_attn.linear_k.bias_scale",
       2.6077429993165424e-06f},
      {"encoder.encoders.30.self_attn.linear_q.bias_scale",
       2.1882689907215536e-05f},
      {"encoder.encoders.30.self_attn.linear_v.bias_scale",
       3.381683882253128e-06f},
      {"/linear_k_31/Add_output_0_scale", 0.00020513647177722305f},
      {"/linear_k_31/MatMul_output_0_scale", 0.00020638405112549663f},
      {"/linear_k_31/Transpose_output_0_scale", 1.6365449482691474e-05f},
      {"/linear_q_31/Add_output_0_scale", 0.00021674142044503242f},
      {"/linear_q_31/MatMul_output_0_scale", 0.00020594164379872382f},
      {"/linear_q_31/Transpose_output_0_scale", 1.4540029951604083e-05f},
      {"/linear_v_31/Add_output_0_scale", 0.00016046759265009314f},
      {"/linear_v_31/MatMul_output_0_scale", 0.0001608187158126384f},
      {"/linear_v_31/Transpose_output_0_scale", 1.2465011423046235e-05f},
      {"/norm1_31/Add_1_output_0_scale", 0.00017365140956826508f},
      {"encoder.encoders.31.self_attn.linear_k.bias_scale",
       5.0068665586877614e-06f},
      {"encoder.encoders.31.self_attn.linear_q.bias_scale",
       2.1897591068409383e-05f},
      {"encoder.encoders.31.self_attn.linear_v.bias_scale",
       4.485318186198128e-06f},
      {"/linear_k_32/Add_output_0_scale", 0.0002127136685885489f},
      {"/linear_k_32/MatMul_output_0_scale", 0.00021378221572376788f},
      {"/linear_k_32/Transpose_output_0_scale", 1.613447784620803e-05f},
      {"/linear_q_32/Add_output_0_scale", 0.0003678415378089994f},
      {"/linear_q_32/MatMul_output_0_scale", 0.0003728681767825037f},
      {"/linear_q_32/Transpose_output_0_scale", 1.3213806596468203e-05f},
      {"/linear_v_32/Add_output_0_scale", 0.00019485375378280878f},
      {"/linear_v_32/MatMul_output_0_scale", 0.00019554201571736485f},
      {"/linear_v_32/Transpose_output_0_scale", 1.326223627984291e-05f},
      {"/norm1_32/Add_1_output_0_scale", 0.00018678793276194483f},
      {"encoder.encoders.32.self_attn.linear_k.bias_scale",
       3.6033420656167436e-06f},
      {"encoder.encoders.32.self_attn.linear_q.bias_scale",
       2.4706501790205948e-05f},
      {"encoder.encoders.32.self_attn.linear_v.bias_scale",
       2.9774837457807735e-06f},
      {"/linear_k_33/Add_output_0_scale", 0.00025845563504844904f},
      {"/linear_k_33/MatMul_output_0_scale", 0.0002573205856606364f},
      {"/linear_k_33/Transpose_output_0_scale", 1.6566618796787225e-05f},
      {"/linear_q_33/Add_output_0_scale", 0.0002820826484821737f},
      {"/linear_q_33/MatMul_output_0_scale", 0.000275878090178594f},
      {"/linear_q_33/Transpose_output_0_scale", 1.2900876754429191e-05f},
      {"/linear_v_33/Add_output_0_scale", 0.00024570964160375297f},
      {"/linear_v_33/MatMul_output_0_scale", 0.00024606529041193426f},
      {"/linear_v_33/Transpose_output_0_scale", 1.4748648936802056e-05f},
      {"/norm1_33/Add_1_output_0_scale", 0.00019120320212095976f},
      {"encoder.encoders.33.self_attn.linear_k.bias_scale",
       6.213879260030808e-06f},
      {"encoder.encoders.33.self_attn.linear_q.bias_scale",
       2.2247773813433014e-05f},
      {"encoder.encoders.33.self_attn.linear_v.bias_scale",
       3.1730644423078047e-06f},
      {"/linear_k_34/Add_output_0_scale", 0.00025653812917880714f},
      {"/linear_k_34/MatMul_output_0_scale", 0.0002524085866753012f},
      {"/linear_k_34/Transpose_output_0_scale", 1.7188751371577382e-05f},
      {"/linear_q_34/Add_output_0_scale", 0.000267638242803514f},
      {"/linear_q_34/MatMul_output_0_scale", 0.0002606476191431284f},
      {"/linear_q_34/Transpose_output_0_scale", 1.1436815839260817e-05f},
      {"/linear_v_34/Add_output_0_scale", 0.00028763985028490424f},
      {"/linear_v_34/MatMul_output_0_scale", 0.0002870669704861939f},
      {"/linear_v_34/Transpose_output_0_scale", 1.7509131794213317e-05f},
      {"/norm1_34/Add_1_output_0_scale", 0.00020622010924853384f},
      {"encoder.encoders.34.self_attn.linear_k.bias_scale",
       7.074434051901335e-06f},
      {"encoder.encoders.34.self_attn.linear_q.bias_scale",
       2.4035940441535786e-05f},
      {"encoder.encoders.34.self_attn.linear_v.bias_scale",
       2.861066604964435e-06f},
      {"/linear_k_35/Add_output_0_scale", 0.00024195743026211858f},
      {"/linear_k_35/MatMul_output_0_scale", 0.00022113832528702915f},
      {"/linear_k_35/Transpose_output_0_scale", 1.6242513083852828e-05f},
      {"/linear_q_35/Add_output_0_scale", 0.00026628468185663223f},
      {"/linear_q_35/MatMul_output_0_scale", 0.00025883212219923735f},
      {"/linear_q_35/Transpose_output_0_scale", 1.2323448572715279e-05f},
      {"/linear_v_35/Add_output_0_scale", 0.0003085963544435799f},
      {"/linear_v_35/MatMul_output_0_scale", 0.00030931513174436986f},
      {"/linear_v_35/Transpose_output_0_scale", 2.03850995603716e-05f},
      {"/norm1_35/Add_1_output_0_scale", 0.00020303847850300372f},
      {"encoder.encoders.35.self_attn.linear_k.bias_scale",
       6.31967923254706e-05f},
      {"encoder.encoders.35.self_attn.linear_q.bias_scale",
       1.8410664779366925e-05f},
      {"encoder.encoders.35.self_attn.linear_v.bias_scale",
       1.8528945702200872e-06f},
      {"/MatMul_2_output_0_scale", 3.177989856339991e-05f},
      {"/linear_out/Add_output_0_scale", 5.36573825229425e-05f},
      {"/linear_out/MatMul_output_0_scale", 4.785883356817067e-05f},
      {"/linear_out/Transpose_output_0_scale", 2.4289263819810003e-05f},
      {"encoder.encoders.0.self_attn.linear_out.bias_scale",
       1.322870775766205e-05f},
      {"/MatMul_5_output_0_scale", 4.210516271996312e-05f},
      {"/linear_out_1/Add_output_0_scale", 5.3722207667306066e-05f},
      {"/linear_out_1/MatMul_output_0_scale", 5.108076220494695e-05f},
      {"/linear_out_1/Transpose_output_0_scale", 1.8500073565519415e-05f},
      {"encoder.encoders.1.self_attn.linear_out.bias_scale",
       1.0103141903528012e-05f},
      {"/MatMul_8_output_0_scale", 5.559751662076451e-05f},
      {"/linear_out_2/Add_output_0_scale", 7.218676182674244e-05f},
      {"/linear_out_2/MatMul_output_0_scale", 7.694277883274481e-05f},
      {"/linear_out_2/Transpose_output_0_scale", 1.753148353600409e-05f},
      {"encoder.encoders.2.self_attn.linear_out.bias_scale",
       1.39104458867223e-05f},
      {"/MatMul_11_output_0_scale", 3.928698424715549e-05f},
      {"/linear_out_3/Add_output_0_scale", 0.00010948812996502966f},
      {"/linear_out_3/MatMul_output_0_scale", 0.00011084206198574975f},
      {"/linear_out_3/Transpose_output_0_scale", 2.498217872926034e-05f},
      {"encoder.encoders.3.self_attn.linear_out.bias_scale",
       7.2756029112497345e-06f},
      {"/MatMul_14_output_0_scale", 7.229391485452652e-05f},
      {"/linear_out_4/Add_output_0_scale", 7.715412357356399e-05f},
      {"/linear_out_4/MatMul_output_0_scale", 7.913856825325638e-05f},
      {"/linear_out_4/Transpose_output_0_scale", 2.103330916725099e-05f},
      {"encoder.encoders.4.self_attn.linear_out.bias_scale",
       9.700804184831213e-06f},
      {"/MatMul_17_output_0_scale", 6.839355773990974e-05f},
      {"/linear_out_5/Add_output_0_scale", 0.0001505417749285698f},
      {"/linear_out_5/MatMul_output_0_scale", 0.00015069544315338135f},
      {"/linear_out_5/Transpose_output_0_scale", 2.121212673955597e-05f},
      {"encoder.encoders.5.self_attn.linear_out.bias_scale",
       6.923557521076873e-06f},
      {"/MatMul_20_output_0_scale", 4.3059917516075075e-05f},
      {"/linear_out_6/Add_output_0_scale", 0.00013693467190023512f},
      {"/linear_out_6/MatMul_output_0_scale", 0.00013658333045896143f},
      {"/linear_out_6/Transpose_output_0_scale", 1.7546384697197936e-05f},
      {"encoder.encoders.6.self_attn.linear_out.bias_scale",
       6.444850441766903e-06f},
      {"/MatMul_23_output_0_scale", 7.773197285132483e-05f},
      {"/linear_out_7/Add_output_0_scale", 0.00013001210754737258f},
      {"/linear_out_7/MatMul_output_0_scale", 0.00012966518988832831f},
      {"/linear_out_7/Transpose_output_0_scale", 2.0332945496193133e-05f},
      {"encoder.encoders.7.self_attn.linear_out.bias_scale",
       7.351972726610256e-06f},
      {"/MatMul_26_output_0_scale", 0.00010629077587509528f},
      {"/linear_out_8/Add_output_0_scale", 0.0002238756133010611f},
      {"/linear_out_8/MatMul_output_0_scale", 0.00021512787498068064f},
      {"/linear_out_8/Transpose_output_0_scale", 1.5612929928465746e-05f},
      {"encoder.encoders.8.self_attn.linear_out.bias_scale",
       1.66672034538351e-05f},
      {"/MatMul_29_output_0_scale", 8.407253335462883e-05f},
      {"/linear_out_9/Add_output_0_scale", 0.00014576951798517257f},
      {"/linear_out_9/MatMul_output_0_scale", 0.00014576951798517257f},
      {"/linear_out_9/Transpose_output_0_scale", 3.1926225346978754e-05f},
      {"encoder.encoders.9.self_attn.linear_out.bias_scale",
       1.1436815839260817e-05f},
      {"/MatMul_32_output_0_scale", 7.157396612456068e-05f},
      {"/linear_out_10/Add_output_0_scale", 0.00014037135406397283f},
      {"/linear_out_10/MatMul_output_0_scale", 0.00013864185893908143f},
      {"/linear_out_10/Transpose_output_0_scale", 1.9789044017670676e-05f},
      {"encoder.encoders.10.self_attn.linear_out.bias_scale",
       1.311694722971879e-05f},
      {"/MatMul_35_output_0_scale", 7.497709157178178e-05f},
      {"/linear_out_11/Add_output_0_scale", 0.00015004383749328554f},
      {"/linear_out_11/MatMul_output_0_scale", 0.00015374031499959528f},
      {"/linear_out_11/Transpose_output_0_scale", 2.0027466234751046e-05f},
      {"encoder.encoders.11.self_attn.linear_out.bias_scale",
       1.0389992894488387e-05f},
      {"/MatMul_38_output_0_scale", 8.524123404640704e-05f},
      {"/linear_out_12/Add_output_0_scale", 0.00020123965805396438f},
      {"/linear_out_12/MatMul_output_0_scale", 0.00020123965805396438f},
      {"/linear_out_12/Transpose_output_0_scale", 2.1197225578362122e-05f},
      {"encoder.encoders.12.self_attn.linear_out.bias_scale",
       2.1556721549131908e-05f},
      {"/MatMul_41_output_0_scale", 9.67684100032784e-05f},
      {"/linear_out_13/Add_output_0_scale", 0.00016586403944529593f},
      {"/linear_out_13/MatMul_output_0_scale", 0.00016586403944529593f},
      {"/linear_out_13/Transpose_output_0_scale", 3.6031557101523504e-05f},
      {"encoder.encoders.13.self_attn.linear_out.bias_scale",
       1.5690231521148235e-05f},
      {"/MatMul_44_output_0_scale", 9.04509870451875e-05f},
      {"/linear_out_14/Add_output_0_scale", 0.00010881401976803318f},
      {"/linear_out_14/MatMul_output_0_scale", 0.00010959591600112617f},
      {"/linear_out_14/Transpose_output_0_scale", 1.7646969354245812e-05f},
      {"encoder.encoders.14.self_attn.linear_out.bias_scale",
       7.036249371594749e-06f},
      {"/MatMul_47_output_0_scale", 7.248138717841357e-05f},
      {"/linear_out_15/Add_output_0_scale", 7.557080243714154e-05f},
      {"/linear_out_15/MatMul_output_0_scale", 7.488580740755424e-05f},
      {"/linear_out_15/Transpose_output_0_scale", 1.215953307109885e-05f},
      {"encoder.encoders.15.self_attn.linear_out.bias_scale",
       3.9479364204453304e-06f},
      {"/MatMul_50_output_0_scale", 7.2657254349906e-05f},
      {"/linear_out_16/Add_output_0_scale", 8.816352055873722e-05f},
      {"/linear_out_16/MatMul_output_0_scale", 8.857751527102664e-05f},
      {"/linear_out_16/Transpose_output_0_scale", 1.0423521416669246e-05f},
      {"encoder.encoders.16.self_attn.linear_out.bias_scale",
       2.1476625988725573e-06f},
      {"/MatMul_53_output_0_scale", 7.997539796633646e-05f},
      {"/linear_out_17/Add_output_0_scale", 0.00012583246279973537f},
      {"/linear_out_17/MatMul_output_0_scale", 0.00012583246279973537f},
      {"/linear_out_17/Transpose_output_0_scale", 1.252834226761479e-05f},
      {"encoder.encoders.17.self_attn.linear_out.bias_scale",
       2.3013330974208657e-06f},
      {"/MatMul_56_output_0_scale", 8.507253369316459e-05f},
      {"/linear_out_18/Add_output_0_scale", 9.287665307056159e-05f},
      {"/linear_out_18/MatMul_output_0_scale", 9.287665307056159e-05f},
      {"/linear_out_18/Transpose_output_0_scale", 1.063586569216568e-05f},
      {"encoder.encoders.18.self_attn.linear_out.bias_scale",
       2.8470965389715275e-06f},
      {"/MatMul_59_output_0_scale", 0.0001412935962434858f},
      {"/linear_out_19/Add_output_0_scale", 0.00011323752551106736f},
      {"/linear_out_19/MatMul_output_0_scale", 0.00011527649621712044f},
      {"/linear_out_19/Transpose_output_0_scale", 1.264010279555805e-05f},
      {"encoder.encoders.19.self_attn.linear_out.bias_scale",
       3.6182434541842667e-06f},
      {"/MatMul_62_output_0_scale", 9.304084233008325e-05f},
      {"/linear_out_20/Add_output_0_scale", 0.0001207901441375725f},
      {"/linear_out_20/MatMul_output_0_scale", 0.00011820661165984347f},
      {"/linear_out_20/Transpose_output_0_scale", 1.2152082490501925e-05f},
      {"encoder.encoders.20.self_attn.linear_out.bias_scale",
       4.241307578922715e-06f},
      {"/MatMul_65_output_0_scale", 0.00010341947199776769f},
      {"/linear_out_21/Add_output_0_scale", 9.177286847261712e-05f},
      {"/linear_out_21/MatMul_output_0_scale", 9.146381489699706e-05f},
      {"/linear_out_21/Transpose_output_0_scale", 1.1123886906716507e-05f},
      {"encoder.encoders.21.self_attn.linear_out.bias_scale",
       4.826187250728253e-06f},
      {"/MatMul_68_output_0_scale", 9.45523424888961e-05f},
      {"/linear_out_22/Add_output_0_scale", 0.00010433581337565556f},
      {"/linear_out_22/MatMul_output_0_scale", 0.00010730261419666931f},
      {"/linear_out_22/Transpose_output_0_scale", 1.5005698514869437e-05f},
      {"encoder.encoders.22.self_attn.linear_out.bias_scale",
       5.405478532338748e-06f},
      {"/MatMul_71_output_0_scale", 7.244919106597081e-05f},
      {"/linear_out_23/Add_output_0_scale", 0.00013481058704201132f},
      {"/linear_out_23/MatMul_output_0_scale", 0.00013481058704201132f},
      {"/linear_out_23/Transpose_output_0_scale", 1.1775822713389061e-05f},
      {"encoder.encoders.23.self_attn.linear_out.bias_scale",
       4.8029037316155154e-06f},
      {"/MatMul_74_output_0_scale", 0.00010644466237863526f},
      {"/linear_out_24/Add_output_0_scale", 0.00011019533849321306f},
      {"/linear_out_24/MatMul_output_0_scale", 0.00011019533849321306f},
      {"/linear_out_24/Transpose_output_0_scale", 1.142563996836543e-05f},
      {"encoder.encoders.24.self_attn.linear_out.bias_scale",
       5.928890004724963e-06f},
      {"/MatMul_77_output_0_scale", 8.931208867579699e-05f},
      {"/linear_out_25/Add_output_0_scale", 8.783922385191545e-05f},
      {"/linear_out_25/MatMul_output_0_scale", 8.442212856607512e-05f},
      {"/linear_out_25/Transpose_output_0_scale", 1.0948795534204692e-05f},
      {"encoder.encoders.25.self_attn.linear_out.bias_scale",
       7.622060365974903e-06f},
      {"/MatMul_80_output_0_scale", 0.00010214641224592924f},
      {"/linear_out_26/Add_output_0_scale", 0.00013415883586276323f},
      {"/linear_out_26/MatMul_output_0_scale", 0.00013415883586276323f},
      {"/linear_out_26/Transpose_output_0_scale", 1.2573046660691034e-05f},
      {"encoder.encoders.26.self_attn.linear_out.bias_scale",
       7.109825219231425e-06f},
      {"/MatMul_83_output_0_scale", 9.056699491338804e-05f},
      {"/linear_out_27/Add_output_0_scale", 0.0001605976140126586f},
      {"/linear_out_27/MatMul_output_0_scale", 0.0001605976140126586f},
      {"/linear_out_27/Transpose_output_0_scale", 1.9919430997106247e-05f},
      {"encoder.encoders.27.self_attn.linear_out.bias_scale",
       6.780132025596686e-06f},
      {"/MatMul_86_output_0_scale", 8.650579547975212e-05f},
      {"/linear_out_28/Add_output_0_scale", 0.00015308371803257614f},
      {"/linear_out_28/MatMul_output_0_scale", 0.00015308371803257614f},
      {"/linear_out_28/Transpose_output_0_scale", 1.5020599676063284e-05f},
      {"encoder.encoders.28.self_attn.linear_out.bias_scale",
       6.634843430219917e-06f},
      {"/MatMul_89_output_0_scale", 9.365901496494189e-05f},
      {"/linear_out_29/Add_output_0_scale", 0.00012700828665401787f},
      {"/linear_out_29/MatMul_output_0_scale", 0.00012347623123787344f},
      {"/linear_out_29/Transpose_output_0_scale", 1.2736962162307464e-05f},
      {"encoder.encoders.29.self_attn.linear_out.bias_scale",
       6.411322374333395e-06f},
      {"/MatMul_92_output_0_scale", 0.00012035842519253492f},
      {"/linear_out_30/Add_output_0_scale", 0.0002181091404054314f},
      {"/linear_out_30/MatMul_output_0_scale", 0.0002181091404054314f},
      {"/linear_out_30/Transpose_output_0_scale", 1.5501169400522485e-05f},
      {"encoder.encoders.30.self_attn.linear_out.bias_scale",
       7.784113222442102e-06f},
      {"/MatMul_95_output_0_scale", 0.00014797982294112444f},
      {"/linear_out_31/Add_output_0_scale", 0.00023604072339367121f},
      {"/linear_out_31/MatMul_output_0_scale", 0.00023604072339367121f},
      {"/linear_out_31/Transpose_output_0_scale", 1.5676261682529002e-05f},
      {"encoder.encoders.31.self_attn.linear_out.bias_scale",
       7.804602319083642e-06f},
      {"/MatMul_98_output_0_scale", 0.00016303054871968925f},
      {"/linear_out_32/Add_output_0_scale", 0.000285319983959198f},
      {"/linear_out_32/MatMul_output_0_scale", 0.000285319983959198f},
      {"/linear_out_32/Transpose_output_0_scale", 2.3372827854473144e-05f},
      {"encoder.encoders.32.self_attn.linear_out.bias_scale",
       1.7393645975971594e-05f},
      {"/MatMul_101_output_0_scale", 0.00018527053180150688f},
      {"/linear_out_33/Add_output_0_scale", 0.0004712661902885884f},
      {"/linear_out_33/MatMul_output_0_scale", 0.0004712661902885884f},
      {"/linear_out_33/Transpose_output_0_scale", 2.4557488359278068e-05f},
      {"encoder.encoders.33.self_attn.linear_out.bias_scale",
       2.1316436686902307e-05f},
      {"/MatMul_104_output_0_scale", 0.0002481398987583816f},
      {"/linear_out_34/Add_output_0_scale", 0.0007127674762159586f},
      {"/linear_out_34/MatMul_output_0_scale", 0.0007127674762159586f},
      {"/linear_out_34/Transpose_output_0_scale", 2.853615842468571e-05f},
      {"encoder.encoders.34.self_attn.linear_out.bias_scale",
       2.3797518224455416e-05f},
      {"/MatMul_107_output_0_scale", 0.00023818970657885075f},
      {"/linear_out_35/Add_output_0_scale", 0.0008459921809844673f},
      {"/linear_out_35/MatMul_output_0_scale", 0.0008459921809844673f},
      {"/linear_out_35/Transpose_output_0_scale", 2.2255224394029938e-05f},
      {"encoder.encoders.35.self_attn.linear_out.bias_scale",
       1.237932883668691e-05f},
      {"/feed_forward/act/Relu_output_0_scale", 1.7403721358277835e-05f},
      {"/feed_forward/w_1/MatMul_output_0_scale", 3.486627610982396e-05f},
      {"/feed_forward/w_1/Transpose_output_0_scale", 2.7776188289863057e-05f},
      {"/feed_forward/w_2/Add_output_0_scale", 0.00010161630780203268f},
      {"/feed_forward/w_2/MatMul_output_0_scale", 0.00010507342813070863f},
      {"/feed_forward/w_2/Transpose_output_0_scale", 6.492534885182977e-05f},
      {"/norm2/Add_1_output_0_scale", 1.2230225365783554e-05f},
      {"encoder.encoders.0.feed_forward.w_1.bias_scale",
       1.9362491912033875e-06f},
      {"encoder.encoders.0.feed_forward.w_2.bias_scale",
       1.1675238056341186e-05f},
      {"/feed_forward/act_1/Relu_output_0_scale", 2.0073966879863292e-05f},
      {"/feed_forward/w_1_1/MatMul_output_0_scale", 4.259160050423816e-05f},
      {"/feed_forward/w_1_1/Transpose_output_0_scale", 2.779854003165383e-05f},
      {"/feed_forward/w_2_1/Add_output_0_scale", 0.00038769570528529584f},
      {"/feed_forward/w_2_1/MatMul_output_0_scale", 0.00039027270395308733f},
      {"/feed_forward/w_2_1/Transpose_output_0_scale", 5.763857188867405e-05f},
      {"/norm2_1/Add_1_output_0_scale", 1.2317340406298172e-05f},
      {"encoder.encoders.1.feed_forward.w_1.bias_scale",
       2.4568664684920805e-06f},
      {"encoder.encoders.1.feed_forward.w_2.bias_scale",
       1.0054712220153306e-05f},
      {"/feed_forward/act_2/Relu_output_0_scale", 1.9095301468041725e-05f},
      {"/feed_forward/w_1_2/MatMul_output_0_scale", 4.58637805422768e-05f},
      {"/feed_forward/w_1_2/Transpose_output_0_scale", 2.9996495868545026e-05f},
      {"/feed_forward/w_2_2/Add_output_0_scale", 0.0001329910010099411f},
      {"/feed_forward/w_2_2/MatMul_output_0_scale", 0.00013295654207468033f},
      {"/feed_forward/w_2_2/Transpose_output_0_scale", 3.1672901968704537e-05f},
      {"/norm2_2/Add_1_output_0_scale", 1.1959211406065151e-05f},
      {"encoder.encoders.2.feed_forward.w_1.bias_scale",
       1.7183164118250716e-06f},
      {"encoder.encoders.2.feed_forward.w_2.bias_scale",
       4.902557066088775e-06f},
      {"/feed_forward/act_3/Relu_output_0_scale", 2.4207705791923217e-05f},
      {"/feed_forward/w_1_3/MatMul_output_0_scale", 5.046412843512371e-05f},
      {"/feed_forward/w_1_3/Transpose_output_0_scale", 3.302147888462059e-05f},
      {"/feed_forward/w_2_3/Add_output_0_scale", 0.00010432601266074926f},
      {"/feed_forward/w_2_3/MatMul_output_0_scale", 0.00010512416338315234f},
      {"/feed_forward/w_2_3/Transpose_output_0_scale", 2.542921902204398e-05f},
      {"/norm2_3/Add_1_output_0_scale", 1.5730229279142804e-05f},
      {"encoder.encoders.3.feed_forward.w_1.bias_scale",
       2.808911858664942e-06f},
      {"encoder.encoders.3.feed_forward.w_2.bias_scale",
       7.6127466854813974e-06f},
      {"/feed_forward/act_4/Relu_output_0_scale", 4.5217970182420686e-05f},
      {"/feed_forward/w_1_4/MatMul_output_0_scale", 8.245015487773344e-05f},
      {"/feed_forward/w_1_4/Transpose_output_0_scale", 2.9989045287948102e-05f},
      {"/feed_forward/w_2_4/Add_output_0_scale", 0.0002299039624631405f},
      {"/feed_forward/w_2_4/MatMul_output_0_scale", 0.0002329464041395113f},
      {"/feed_forward/w_2_4/Transpose_output_0_scale", 3.0078452255111188e-05f},
      {"/norm2_4/Add_1_output_0_scale", 1.4970222764532082e-05f},
      {"encoder.encoders.4.feed_forward.w_1.bias_scale",
       2.7530315946933115e-06f},
      {"encoder.encoders.4.feed_forward.w_2.bias_scale", 4.56355019196053e-06f},
      {"/feed_forward/act_5/Relu_output_0_scale", 2.0706902432721108e-05f},
      {"/feed_forward/w_1_5/MatMul_output_0_scale", 5.280131153995171e-05f},
      {"/feed_forward/w_1_5/Transpose_output_0_scale", 3.508531881379895e-05f},
      {"/feed_forward/w_2_5/Add_output_0_scale", 0.00034316806704737246f},
      {"/feed_forward/w_2_5/MatMul_output_0_scale", 0.0003472957469057292f},
      {"/feed_forward/w_2_5/Transpose_output_0_scale", 3.341636329423636e-05f},
      {"/norm2_5/Add_1_output_0_scale", 1.453207732993178e-05f},
      {"encoder.encoders.5.feed_forward.w_1.bias_scale",
       3.5036889585171593e-06f},
      {"encoder.encoders.5.feed_forward.w_2.bias_scale",
       6.074178600101732e-06f},
      {"/feed_forward/act_6/Relu_output_0_scale", 4.156336945015937e-05f},
      {"/feed_forward/w_1_6/MatMul_output_0_scale", 8.718191384105012e-05f},
      {"/feed_forward/w_1_6/Transpose_output_0_scale", 3.813265357166529e-05f},
      {"/feed_forward/w_2_6/Add_output_0_scale", 0.0009611161658540368f},
      {"/feed_forward/w_2_6/MatMul_output_0_scale", 0.0009636401082389057f},
      {"/feed_forward/w_2_6/Transpose_output_0_scale", 4.325872941990383e-05f},
      {"/norm2_6/Add_1_output_0_scale", 1.6712918295525014e-05f},
      {"encoder.encoders.6.feed_forward.w_1.bias_scale",
       3.087381401201128e-06f},
      {"encoder.encoders.6.feed_forward.w_2.bias_scale",
       6.336815658869455e-06f},
      {"/feed_forward/act_7/Relu_output_0_scale", 5.188928116695024e-05f},
      {"/feed_forward/w_1_7/MatMul_output_0_scale", 0.00010283762094331905f},
      {"/feed_forward/w_1_7/Transpose_output_0_scale", 3.317049049655907e-05f},
      {"/feed_forward/w_2_7/Add_output_0_scale", 0.0010466875974088907f},
      {"/feed_forward/w_2_7/MatMul_output_0_scale", 0.0010489833075553179f},
      {"/feed_forward/w_2_7/Transpose_output_0_scale", 6.610256241401657e-05f},
      {"/norm2_7/Add_1_output_0_scale", 2.4392129489569925e-05f},
      {"encoder.encoders.7.feed_forward.w_1.bias_scale",
       3.0510593660437735e-06f},
      {"encoder.encoders.7.feed_forward.w_2.bias_scale",
       7.365011242654873e-06f},
      {"/feed_forward/act_8/Relu_output_0_scale", 0.000199572867131792f},
      {"/feed_forward/w_1_8/MatMul_output_0_scale", 0.00031136622419580817f},
      {"/feed_forward/w_1_8/Transpose_output_0_scale", 5.364499884308316e-05f},
      {"/feed_forward/w_2_8/Add_output_0_scale", 0.005356259644031525f},
      {"/feed_forward/w_2_8/MatMul_output_0_scale", 0.005357983987778425f},
      {"/feed_forward/w_2_8/Transpose_output_0_scale", 0.00010168707376578823f},
      {"/norm2_8/Add_1_output_0_scale", 7.828785601304844e-05f},
      {"encoder.encoders.8.feed_forward.w_1.bias_scale",
       3.934897904400714e-06f},
      {"encoder.encoders.8.feed_forward.w_2.bias_scale",
       8.911030818126164e-06f},
      {"/feed_forward/act_9/Relu_output_0_scale", 0.00013951834989711642f},
      {"/feed_forward/w_1_9/MatMul_output_0_scale", 0.00022525910753756762f},
      {"/feed_forward/w_1_9/Transpose_output_0_scale", 6.219839269760996e-05f},
      {"/feed_forward/w_2_9/Add_output_0_scale", 0.0021969093941152096f},
      {"/feed_forward/w_2_9/MatMul_output_0_scale", 0.002201005583629012f},
      {"/feed_forward/w_2_9/Transpose_output_0_scale", 0.00011092593922512606f},
      {"/norm2_9/Add_1_output_0_scale", 7.261713471962139e-05f},
      {"encoder.encoders.9.feed_forward.w_1.bias_scale",
       4.2208184822811745e-06f},
      {"encoder.encoders.9.feed_forward.w_2.bias_scale",
       6.707487500534626e-06f},
      {"/feed_forward/act_10/Relu_output_0_scale", 0.0002472416963428259f},
      {"/feed_forward/w_1_10/MatMul_output_0_scale", 0.00042017942178063095f},
      {"/feed_forward/w_1_10/Transpose_output_0_scale", 8.091454219538718e-05f},
      {"/feed_forward/w_2_10/Add_output_0_scale", 0.004754376132041216f},
      {"/feed_forward/w_2_10/MatMul_output_0_scale", 0.004759002942591906f},
      {"/feed_forward/w_2_10/Transpose_output_0_scale",
       0.00013746530748903751f},
      {"/norm2_10/Add_1_output_0_scale", 9.142374619841576e-05f},
      {"encoder.encoders.10.feed_forward.w_1.bias_scale",
       3.879948963003699e-06f},
      {"encoder.encoders.10.feed_forward.w_2.bias_scale",
       8.404383152083028e-06f},
      {"/feed_forward/act_11/Relu_output_0_scale", 0.0003553667920641601f},
      {"/feed_forward/w_1_11/MatMul_output_0_scale", 0.0005647346843034029f},
      {"/feed_forward/w_1_11/Transpose_output_0_scale",
       0.00010463754733791575f},
      {"/feed_forward/w_2_11/Add_output_0_scale", 0.00556331779807806f},
      {"/feed_forward/w_2_11/MatMul_output_0_scale", 0.005565843544900417f},
      {"/feed_forward/w_2_11/Transpose_output_0_scale",
       0.00015262002125382423f},
      {"/norm2_11/Add_1_output_0_scale", 0.00010468094842508435f},
      {"encoder.encoders.11.feed_forward.w_1.bias_scale",
       5.217348643782316e-06f},
      {"encoder.encoders.11.feed_forward.w_2.bias_scale",
       7.826954060874414e-06f},
      {"/feed_forward/act_12/Relu_output_0_scale", 0.00035766439395956695f},
      {"/feed_forward/w_1_12/MatMul_output_0_scale", 0.0005349895800463855f},
      {"/feed_forward/w_1_12/Transpose_output_0_scale",
       0.00010699196718633175f},
      {"/feed_forward/w_2_12/Add_output_0_scale", 0.006595155224204063f},
      {"/feed_forward/w_2_12/MatMul_output_0_scale", 0.006595590617507696f},
      {"/feed_forward/w_2_12/Transpose_output_0_scale", 0.0001333674299530685f},
      {"/norm2_12/Add_1_output_0_scale", 0.00010570665472187102f},
      {"encoder.encoders.12.feed_forward.w_1.bias_scale",
       5.381264145398745e-06f},
      {"encoder.encoders.12.feed_forward.w_2.bias_scale",
       1.037509173329454e-05f},
      {"/feed_forward/act_13/Relu_output_0_scale", 0.0003103148192167282f},
      {"/feed_forward/w_1_13/MatMul_output_0_scale", 0.0004578372754622251f},
      {"/feed_forward/w_1_13/Transpose_output_0_scale",
       0.00010284938616678119f},
      {"/feed_forward/w_2_13/Add_output_0_scale", 0.0042994096875190735f},
      {"/feed_forward/w_2_13/MatMul_output_0_scale", 0.004299695137888193f},
      {"/feed_forward/w_2_13/Transpose_output_0_scale", 0.0001292472006753087f},
      {"/norm2_13/Add_1_output_0_scale", 0.00010070233838632703f},
      {"encoder.encoders.13.feed_forward.w_1.bias_scale",
       2.7902849524252815e-06f},
      {"encoder.encoders.13.feed_forward.w_2.bias_scale",
       5.995946139591979e-06f},
      {"/feed_forward/act_14/Relu_output_0_scale", 0.00032985073630698025f},
      {"/feed_forward/w_1_14/MatMul_output_0_scale", 0.0005408381693996489f},
      {"/feed_forward/w_1_14/Transpose_output_0_scale",
       0.00011632023961283267f},
      {"/feed_forward/w_2_14/Add_output_0_scale", 0.003148949472233653f},
      {"/feed_forward/w_2_14/MatMul_output_0_scale", 0.0031477787997573614f},
      {"/feed_forward/w_2_14/Transpose_output_0_scale",
       0.00012131220137234777f},
      {"/norm2_14/Add_1_output_0_scale", 8.417766366619617e-05f},
      {"encoder.encoders.14.feed_forward.w_1.bias_scale",
       4.973338491254253e-06f},
      {"encoder.encoders.14.feed_forward.w_2.bias_scale",
       5.4110669225337915e-06f},
      {"/feed_forward/act_15/Relu_output_0_scale", 0.00029688343056477606f},
      {"/feed_forward/w_1_15/MatMul_output_0_scale", 0.0005010146414861083f},
      {"/feed_forward/w_1_15/Transpose_output_0_scale",
       0.00011786998220486566f},
      {"/feed_forward/w_2_15/Add_output_0_scale", 0.002533592749387026f},
      {"/feed_forward/w_2_15/MatMul_output_0_scale", 0.0025329801719635725f},
      {"/feed_forward/w_2_15/Transpose_output_0_scale",
       0.00011832447489723563f},
      {"/norm2_15/Add_1_output_0_scale", 8.138832345139235e-05f},
      {"encoder.encoders.15.feed_forward.w_1.bias_scale",
       5.020836397306994e-06f},
      {"encoder.encoders.15.feed_forward.w_2.bias_scale",
       5.463221441459609e-06f},
      {"/feed_forward/act_16/Relu_output_0_scale", 0.0001981027307920158f},
      {"/feed_forward/w_1_16/MatMul_output_0_scale", 0.0003415450919419527f},
      {"/feed_forward/w_1_16/Transpose_output_0_scale", 9.629277337808162e-05f},
      {"/feed_forward/w_2_16/Add_output_0_scale", 0.0014185482868924737f},
      {"/feed_forward/w_2_16/MatMul_output_0_scale", 0.0014188222121447325f},
      {"/feed_forward/w_2_16/Transpose_output_0_scale",
       0.00010714098607422784f},
      {"/norm2_16/Add_1_output_0_scale", 7.602923142258078e-05f},
      {"encoder.encoders.16.feed_forward.w_1.bias_scale",
       6.482104254246224e-06f},
      {"encoder.encoders.16.feed_forward.w_2.bias_scale",
       7.601570814586012e-06f},
      {"/feed_forward/act_17/Relu_output_0_scale", 0.00014045984426047653f},
      {"/feed_forward/w_1_17/MatMul_output_0_scale", 0.0002356313052587211f},
      {"/feed_forward/w_1_17/Transpose_output_0_scale", 7.164587441366166e-05f},
      {"/feed_forward/w_2_17/Add_output_0_scale", 0.0006520146271213889f},
      {"/feed_forward/w_2_17/MatMul_output_0_scale", 0.000650769448839128f},
      {"/feed_forward/w_2_17/Transpose_output_0_scale", 7.952126179588959e-05f},
      {"/norm2_17/Add_1_output_0_scale", 7.27716032997705e-05f},
      {"encoder.encoders.17.feed_forward.w_1.bias_scale",
       4.773100954480469e-06f},
      {"encoder.encoders.17.feed_forward.w_2.bias_scale",
       7.72637031332124e-06f},
      {"/feed_forward/act_18/Relu_output_0_scale", 6.747175939381123e-05f},
      {"/feed_forward/w_1_18/MatMul_output_0_scale", 0.00012954282283317298f},
      {"/feed_forward/w_1_18/Transpose_output_0_scale",
       3.0160410460666753e-05f},
      {"/feed_forward/w_2_18/Add_output_0_scale", 0.00024008114996831864f},
      {"/feed_forward/w_2_18/MatMul_output_0_scale", 0.00024376284272875637f},
      {"/feed_forward/w_2_18/Transpose_output_0_scale", 4.597078441292979e-05f},
      {"/norm2_18/Add_1_output_0_scale", 7.616168295498937e-05f},
      {"encoder.encoders.18.feed_forward.w_1.bias_scale",
       3.230807351428666e-06f},
      {"encoder.encoders.18.feed_forward.w_2.bias_scale",
       6.537984063470503e-06f},
      {"/feed_forward/act_19/Relu_output_0_scale", 3.81822646886576e-05f},
      {"/feed_forward/w_1_19/MatMul_output_0_scale", 0.00011264234490226954f},
      {"/feed_forward/w_1_19/Transpose_output_0_scale", 1.562038050906267e-05f},
      {"/feed_forward/w_2_19/Add_output_0_scale", 0.00012730545131489635f},
      {"/feed_forward/w_2_19/MatMul_output_0_scale", 0.00012837974645663053f},
      {"/feed_forward/w_2_19/Transpose_output_0_scale", 2.805931399052497e-05f},
      {"/norm2_19/Add_1_output_0_scale", 6.52164890198037e-05f},
      {"encoder.encoders.19.feed_forward.w_1.bias_scale",
       3.089244046350359e-06f},
      {"encoder.encoders.19.feed_forward.w_2.bias_scale",
       5.498612608789699e-06f},
      {"/feed_forward/act_20/Relu_output_0_scale", 5.390445585362613e-05f},
      {"/feed_forward/w_1_20/MatMul_output_0_scale", 0.00012803687423001975f},
      {"/feed_forward/w_1_20/Transpose_output_0_scale", 2.808911813190207e-05f},
      {"/feed_forward/w_2_20/Add_output_0_scale", 0.00015455296670552343f},
      {"/feed_forward/w_2_20/MatMul_output_0_scale", 0.0001573926128912717f},
      {"/feed_forward/w_2_20/Transpose_output_0_scale",
       4.1954859625548124e-05f},
      {"/norm2_20/Add_1_output_0_scale", 6.081126775825396e-05f},
      {"encoder.encoders.20.feed_forward.w_1.bias_scale",
       3.1283602766052354e-06f},
      {"encoder.encoders.20.feed_forward.w_2.bias_scale",
       4.980789071851177e-06f},
      {"/feed_forward/act_21/Relu_output_0_scale", 8.135973621392623e-05f},
      {"/feed_forward/w_1_21/MatMul_output_0_scale", 0.0001543205144116655f},
      {"/feed_forward/w_1_21/Transpose_output_0_scale", 4.038276165374555e-05f},
      {"/feed_forward/w_2_21/Add_output_0_scale", 0.000281108426861465f},
      {"/feed_forward/w_2_21/MatMul_output_0_scale", 0.0002824858820531517f},
      {"/feed_forward/w_2_21/Transpose_output_0_scale", 5.631234671454877e-05f},
      {"/norm2_21/Add_1_output_0_scale", 6.248442514333874e-05f},
      {"encoder.encoders.21.feed_forward.w_1.bias_scale",
       3.3090395845647436e-06f},
      {"encoder.encoders.21.feed_forward.w_2.bias_scale",
       5.7090946938842535e-06f},
      {"/feed_forward/act_22/Relu_output_0_scale", 4.1797069570748135e-05f},
      {"/feed_forward/w_1_22/MatMul_output_0_scale", 0.0001046007892000489f},
      {"/feed_forward/w_1_22/Transpose_output_0_scale",
       2.7619724278338253e-05f},
      {"/feed_forward/w_2_22/Add_output_0_scale", 0.00010919987107627094f},
      {"/feed_forward/w_2_22/MatMul_output_0_scale", 0.00011201389861525968f},
      {"/feed_forward/w_2_22/Transpose_output_0_scale", 2.808911813190207e-05f},
      {"/norm2_22/Add_1_output_0_scale", 5.757332473876886e-05f},
      {"encoder.encoders.22.feed_forward.w_1.bias_scale",
       3.53256041307759e-06f},
      {"encoder.encoders.22.feed_forward.w_2.bias_scale",
       4.8932433855952695e-06f},
      {"/feed_forward/act_23/Relu_output_0_scale", 4.792657637153752e-05f},
      {"/feed_forward/w_1_23/MatMul_output_0_scale", 0.00010974804899888113f},
      {"/feed_forward/w_1_23/Transpose_output_0_scale",
       1.6510739442310296e-05f},
      {"/feed_forward/w_2_23/Add_output_0_scale", 9.024520113598555e-05f},
      {"/feed_forward/w_2_23/MatMul_output_0_scale", 9.219588537234813e-05f},
      {"/feed_forward/w_2_23/Transpose_output_0_scale",
       2.4684150048415177e-05f},
      {"/norm2_23/Add_1_output_0_scale", 5.59592735953629e-05f},
      {"encoder.encoders.23.feed_forward.w_1.bias_scale",
       2.965376324937097e-06f},
      {"encoder.encoders.23.feed_forward.w_2.bias_scale",
       4.956574230163824e-06f},
      {"/feed_forward/act_24/Relu_output_0_scale", 4.9383823352400213e-05f},
      {"/feed_forward/w_1_24/MatMul_output_0_scale", 0.00012606596283148974f},
      {"/feed_forward/w_1_24/Transpose_output_0_scale", 2.590606345620472e-05f},
      {"/feed_forward/w_2_24/Add_output_0_scale", 0.00010086532711284235f},
      {"/feed_forward/w_2_24/MatMul_output_0_scale", 9.971326653612778e-05f},
      {"/feed_forward/w_2_24/Transpose_output_0_scale",
       2.6800147679750808e-05f},
      {"/norm2_24/Add_1_output_0_scale", 6.101336839492433e-05f},
      {"encoder.encoders.24.feed_forward.w_1.bias_scale",
       4.07459856432979e-06f},
      {"encoder.encoders.24.feed_forward.w_2.bias_scale",
       4.714426722784992e-06f},
      {"/feed_forward/act_25/Relu_output_0_scale", 5.219504237174988e-05f},
      {"/feed_forward/w_1_25/MatMul_output_0_scale", 0.00013247721653897315f},
      {"/feed_forward/w_1_25/Transpose_output_0_scale",
       2.7738935386878438e-05f},
      {"/feed_forward/w_2_25/Add_output_0_scale", 0.00012509908992797136f},
      {"/feed_forward/w_2_25/MatMul_output_0_scale", 0.00012696129851974547f},
      {"/feed_forward/w_2_25/Transpose_output_0_scale",
       2.5988021661760285e-05f},
      {"/norm2_25/Add_1_output_0_scale", 6.666983244940639e-05f},
      {"encoder.encoders.25.feed_forward.w_1.bias_scale",
       3.4301133382541593e-06f},
      {"encoder.encoders.25.feed_forward.w_2.bias_scale",
       6.2678964241058566e-06f},
      {"/feed_forward/act_26/Relu_output_0_scale", 7.09152445779182e-05f},
      {"/feed_forward/w_1_26/MatMul_output_0_scale", 0.00016668073658365756f},
      {"/feed_forward/w_1_26/Transpose_output_0_scale", 2.41029956669081e-05f},
      {"/feed_forward/w_2_26/Add_output_0_scale", 0.00012227121624164283f},
      {"/feed_forward/w_2_26/MatMul_output_0_scale", 0.00012402946595102549f},
      {"/feed_forward/w_2_26/Transpose_output_0_scale",
       2.3901828171801753e-05f},
      {"/norm2_26/Add_1_output_0_scale", 7.588349399156868e-05f},
      {"encoder.encoders.26.feed_forward.w_1.bias_scale",
       5.940065875620348e-06f},
      {"encoder.encoders.26.feed_forward.w_2.bias_scale",
       7.1284516707237344e-06f},
      {"/feed_forward/act_27/Relu_output_0_scale", 7.248963083839044e-05f},
      {"/feed_forward/w_1_27/MatMul_output_0_scale", 0.0001869133993750438f},
      {"/feed_forward/w_1_27/Transpose_output_0_scale", 2.669583773240447e-05f},
      {"/feed_forward/w_2_27/Add_output_0_scale", 0.00017048775043804199f},
      {"/feed_forward/w_2_27/MatMul_output_0_scale", 0.000172813655808568f},
      {"/feed_forward/w_2_27/Transpose_output_0_scale", 3.305128120700829e-05f},
      {"/norm2_27/Add_1_output_0_scale", 8.755805174587294e-05f},
      {"encoder.encoders.27.feed_forward.w_1.bias_scale",
       5.396165306592593e-06f},
      {"encoder.encoders.27.feed_forward.w_2.bias_scale",
       6.5696494857547805e-06f},
      {"/feed_forward/act_28/Relu_output_0_scale", 9.748918091645464e-05f},
      {"/feed_forward/w_1_28/MatMul_output_0_scale", 0.0002556682738941163f},
      {"/feed_forward/w_1_28/Transpose_output_0_scale",
       2.5049233954632655e-05f},
      {"/feed_forward/w_2_28/Add_output_0_scale", 0.0001819987955968827f},
      {"/feed_forward/w_2_28/MatMul_output_0_scale", 0.0001887983817141503f},
      {"/feed_forward/w_2_28/Transpose_output_0_scale",
       2.9981594707351178e-05f},
      {"/norm2_28/Add_1_output_0_scale", 0.00010161825775867328f},
      {"encoder.encoders.28.feed_forward.w_1.bias_scale",
       6.239956292120041e-06f},
      {"encoder.encoders.28.feed_forward.w_2.bias_scale",
       9.503360161033925e-06f},
      {"/feed_forward/act_29/Relu_output_0_scale", 0.00013056525494903326f},
      {"/feed_forward/w_1_29/MatMul_output_0_scale", 0.0003343661373946816f},
      {"/feed_forward/w_1_29/Transpose_output_0_scale",
       3.4161432267865166e-05f},
      {"/feed_forward/w_2_29/Add_output_0_scale", 0.00024180117179639637f},
      {"/feed_forward/w_2_29/MatMul_output_0_scale", 0.00024180117179639637f},
      {"/feed_forward/w_2_29/Transpose_output_0_scale",
       2.9333383281482384e-05f},
      {"/norm2_29/Add_1_output_0_scale", 0.0001300891162827611f},
      {"encoder.encoders.29.feed_forward.w_1.bias_scale",
       7.797151738486718e-06f},
      {"encoder.encoders.29.feed_forward.w_2.bias_scale",
       1.542293648526538e-05f},
      {"/feed_forward/act_30/Relu_output_0_scale", 0.00011958463437622413f},
      {"/feed_forward/w_1_30/MatMul_output_0_scale", 0.0003496803983580321f},
      {"/feed_forward/w_1_30/Transpose_output_0_scale",
       2.4937473426689394e-05f},
      {"/feed_forward/w_2_30/Add_output_0_scale", 0.0002968067128676921f},
      {"/feed_forward/w_2_30/MatMul_output_0_scale", 0.00030864865402691066f},
      {"/feed_forward/w_2_30/Transpose_output_0_scale",
       2.7567568395170383e-05f},
      {"/norm2_30/Add_1_output_0_scale", 0.00016166505520232022f},
      {"encoder.encoders.30.feed_forward.w_1.bias_scale",
       7.0483570198121015e-06f},
      {"encoder.encoders.30.feed_forward.w_2.bias_scale",
       1.5240395441651344e-05f},
      {"/feed_forward/act_31/Relu_output_0_scale", 0.0001712484663585201f},
      {"/feed_forward/w_1_31/MatMul_output_0_scale", 0.0004408749518916011f},
      {"/feed_forward/w_1_31/Transpose_output_0_scale", 2.377516466367524e-05f},
      {"/feed_forward/w_2_31/Add_output_0_scale", 0.00043497714796103537f},
      {"/feed_forward/w_2_31/MatMul_output_0_scale", 0.00043497714796103537f},
      {"/feed_forward/w_2_31/Transpose_output_0_scale",
       3.0257269827416167e-05f},
      {"/norm2_31/Add_1_output_0_scale", 0.00019507679098751396f},
      {"encoder.encoders.31.feed_forward.w_1.bias_scale",
       8.359678759006783e-06f},
      {"encoder.encoders.31.feed_forward.w_2.bias_scale",
       1.9368078937986866e-05f},
      {"/feed_forward/act_32/Relu_output_0_scale", 0.00025099533377215266f},
      {"/feed_forward/w_1_32/MatMul_output_0_scale", 0.0005518029211089015f},
      {"/feed_forward/w_1_32/Transpose_output_0_scale", 1.869379229901824e-05f},
      {"/feed_forward/w_2_32/Add_output_0_scale", 0.0005078887334093451f},
      {"/feed_forward/w_2_32/MatMul_output_0_scale", 0.0005078887334093451f},
      {"/feed_forward/w_2_32/Transpose_output_0_scale",
       4.4048505515092984e-05f},
      {"/norm2_32/Add_1_output_0_scale", 0.00023418043565470725f},
      {"encoder.encoders.32.feed_forward.w_1.bias_scale",
       8.439774319413118e-06f},
      {"encoder.encoders.32.feed_forward.w_2.bias_scale",
       2.0459607185330242e-05f},
      {"/feed_forward/act_33/Relu_output_0_scale", 0.0002757632464636117f},
      {"/feed_forward/w_1_33/MatMul_output_0_scale", 0.0006662971572950482f},
      {"/feed_forward/w_1_33/Transpose_output_0_scale", 1.983374750125222e-05f},
      {"/feed_forward/w_2_33/Add_output_0_scale", 0.0007395974243991077f},
      {"/feed_forward/w_2_33/MatMul_output_0_scale", 0.0007395974243991077f},
      {"/feed_forward/w_2_33/Transpose_output_0_scale", 3.765580913750455e-05f},
      {"/norm2_33/Add_1_output_0_scale", 0.0003058896108996123f},
      {"encoder.encoders.33.feed_forward.w_1.bias_scale",
       9.615120688977186e-06f},
      {"encoder.encoders.33.feed_forward.w_2.bias_scale",
       2.1025858586654067e-05f},
      {"/feed_forward/act_34/Relu_output_0_scale", 0.0002262214111397043f},
      {"/feed_forward/w_1_34/MatMul_output_0_scale", 0.0006460344884544611f},
      {"/feed_forward/w_1_34/Transpose_output_0_scale",
       2.2165815607877448e-05f},
      {"/feed_forward/w_2_34/Add_output_0_scale", 0.0008826724370010197f},
      {"/feed_forward/w_2_34/MatMul_output_0_scale", 0.0008826724370010197f},
      {"/feed_forward/w_2_34/Transpose_output_0_scale", 4.050942516187206e-05f},
      {"/norm2_34/Add_1_output_0_scale", 0.0003814825031440705f},
      {"encoder.encoders.34.feed_forward.w_1.bias_scale",
       1.0771841516543645e-05f},
      {"encoder.encoders.34.feed_forward.w_2.bias_scale",
       2.1159972675377503e-05f},
      {"/feed_forward/act_35/Relu_output_0_scale", 0.0005071600317023695f},
      {"/feed_forward/w_1_35/MatMul_output_0_scale", 0.0008586327894590795f},
      {"/feed_forward/w_1_35/Transpose_output_0_scale",
       2.0809789930353872e-05f},
      {"/feed_forward/w_2_35/Add_output_0_scale", 0.0011475899955257773f},
      {"/feed_forward/w_2_35/MatMul_output_0_scale", 0.0011495485669001937f},
      {"/feed_forward/w_2_35/Transpose_output_0_scale", 5.821972445119172e-05f},
      {"/norm2_35/Add_1_output_0_scale", 0.000332554685883224f},
      {"encoder.encoders.35.feed_forward.w_1.bias_scale",
       1.705836439214181e-05f},
      {"encoder.encoders.35.feed_forward.w_2.bias_scale",
       1.1734843610611279e-05f},
      {"/Concat_6_output_0_scale", 8.286539377877489e-05f},
      {"/Mul_output_0_scale", 1.2028699529764708e-05f},
      {"/MatMul_output_0_scale", 0.00027249514823779464f},
      {"/MatMul_1_output_0_scale", 0.00022123918461147696f},
      {"/Add_output_0_scale", 0.00029240373987704515f},
      {"/Slice_1_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_3_output_0_scale", 0.00029240373987704515f},
      {"/ReduceMin_output_0_scale", 0.00012590766709763557f},
      {"/Expand_output_0_scale", 0.00012590766709763557f},
      {"/Tile_output_0_scale", 0.00012590766709763557f},
      {"/Slice_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_4_output_0_scale", 0.00012590766709763557f},
      {"/Add_2_output_0_scale", 0.00029240373987704515f},
      {"/Concat_122_output_0_scale", 0.000124235128168948f},
      {"/Mul_5_output_0_scale", 1.9190723833162338e-05f},
      {"/MatMul_3_output_0_scale", 0.00025493052089586854f},
      {"/MatMul_4_output_0_scale", 0.000294464553007856f},
      {"/Add_5_output_0_scale", 0.00042953863157890737f},
      {"/Slice_5_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_8_output_0_scale", 0.00042953863157890737f},
      {"/ReduceMin_1_output_0_scale", 0.00026457346393726766f},
      {"/Expand_1_output_0_scale", 0.00026457346393726766f},
      {"/Tile_1_output_0_scale", 0.00026457346393726766f},
      {"/Slice_4_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_9_output_0_scale", 0.00026457346393726766f},
      {"/Add_7_output_0_scale", 0.00042953863157890737f},
      {"/Concat_130_output_0_scale", 0.00013973670138511807f},
      {"/Mul_10_output_0_scale", 2.6336691007600166e-05f},
      {"/MatMul_6_output_0_scale", 0.00025164728867821395f},
      {"/MatMul_7_output_0_scale", 0.00023239219444803894f},
      {"/Add_10_output_0_scale", 0.00030680870986543596f},
      {"/Slice_9_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_13_output_0_scale", 0.00030680870986543596f},
      {"/ReduceMin_2_output_0_scale", 0.0001657905668253079f},
      {"/Expand_2_output_0_scale", 0.0001657905668253079f},
      {"/Tile_2_output_0_scale", 0.0001657905668253079f},
      {"/Slice_8_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_14_output_0_scale", 0.0001657905668253079f},
      {"/Add_12_output_0_scale", 0.00030680870986543596f},
      {"/Concat_138_output_0_scale", 9.464300092076883e-05f},
      {"/Mul_15_output_0_scale", 1.77395577338757e-05f},
      {"/MatMul_9_output_0_scale", 0.0002426465362077579f},
      {"/MatMul_10_output_0_scale", 0.0002672488917596638f},
      {"/Add_15_output_0_scale", 0.0003854273818433285f},
      {"/Slice_13_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_18_output_0_scale", 0.00038538724766112864f},
      {"/ReduceMin_3_output_0_scale", 0.00016691372729837894f},
      {"/Expand_3_output_0_scale", 0.00016691372729837894f},
      {"/Tile_3_output_0_scale", 0.00016691372729837894f},
      {"/Slice_12_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_19_output_0_scale", 0.00016691372729837894f},
      {"/Add_17_output_0_scale", 0.00038538724766112864f},
      {"/Concat_146_output_0_scale", 0.00012817306560464203f},
      {"/Mul_20_output_0_scale", 1.3941254110250156e-05f},
      {"/MatMul_12_output_0_scale", 0.0002463336568325758f},
      {"/MatMul_13_output_0_scale", 0.00018324586562812328f},
      {"/Add_20_output_0_scale", 0.00030306633561849594f},
      {"/Slice_17_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_23_output_0_scale", 0.00030306633561849594f},
      {"/ReduceMin_4_output_0_scale", 0.00015092571265995502f},
      {"/Expand_4_output_0_scale", 0.00015092571265995502f},
      {"/Tile_4_output_0_scale", 0.00015092571265995502f},
      {"/Slice_16_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_24_output_0_scale", 0.00015092571265995502f},
      {"/Add_22_output_0_scale", 0.00030306633561849594f},
      {"/Concat_154_output_0_scale", 9.32643742999062e-05f},
      {"/Mul_25_output_0_scale", 2.0708272131741978e-05f},
      {"/MatMul_15_output_0_scale", 0.00022158908541314304f},
      {"/MatMul_16_output_0_scale", 0.00020275723363738507f},
      {"/Add_25_output_0_scale", 0.0003014373069163412f},
      {"/Slice_21_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_28_output_0_scale", 0.0002845364506356418f},
      {"/ReduceMin_5_output_0_scale", 0.00013271084753796458f},
      {"/Expand_5_output_0_scale", 0.00013271084753796458f},
      {"/Tile_5_output_0_scale", 0.00013271084753796458f},
      {"/Slice_20_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_29_output_0_scale", 0.00013271084753796458f},
      {"/Add_27_output_0_scale", 0.0002845364506356418f},
      {"/Concat_162_output_0_scale", 9.256654448108748e-05f},
      {"/Mul_30_output_0_scale", 1.3342031706997659e-05f},
      {"/MatMul_18_output_0_scale", 0.00020770297851413488f},
      {"/MatMul_19_output_0_scale", 0.0001995751663343981f},
      {"/Add_30_output_0_scale", 0.0002748722326941788f},
      {"/Slice_25_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_33_output_0_scale", 0.0002748722326941788f},
      {"/ReduceMin_6_output_0_scale", 0.00012945206253789365f},
      {"/Expand_6_output_0_scale", 0.00012945206253789365f},
      {"/Tile_6_output_0_scale", 0.00012945206253789365f},
      {"/Slice_24_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_34_output_0_scale", 0.00012945206253789365f},
      {"/Add_32_output_0_scale", 0.0002748722326941788f},
      {"/Concat_170_output_0_scale", 0.00010678502439986914f},
      {"/Mul_35_output_0_scale", 1.661268288444262e-05f},
      {"/MatMul_21_output_0_scale", 0.0002497013483662158f},
      {"/MatMul_22_output_0_scale", 0.000178758695255965f},
      {"/Add_35_output_0_scale", 0.0002837470965459943f},
      {"/Slice_29_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_38_output_0_scale", 0.00028293454670347273f},
      {"/ReduceMin_7_output_0_scale", 0.00014978826220612973f},
      {"/Expand_7_output_0_scale", 0.00014978826220612973f},
      {"/Tile_7_output_0_scale", 0.00014978826220612973f},
      {"/Slice_28_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_39_output_0_scale", 0.00014978826220612973f},
      {"/Add_37_output_0_scale", 0.00028293454670347273f},
      {"/Concat_178_output_0_scale", 0.00013839158054906875f},
      {"/Mul_40_output_0_scale", 6.336349906632677e-05f},
      {"/MatMul_24_output_0_scale", 0.00026699143927544355f},
      {"/MatMul_25_output_0_scale", 0.0007405258365906775f},
      {"/Add_40_output_0_scale", 0.0007873804424889386f},
      {"/Slice_33_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_43_output_0_scale", 0.0007873804424889386f},
      {"/ReduceMin_8_output_0_scale", 0.0003696007188409567f},
      {"/Expand_8_output_0_scale", 0.0003696007188409567f},
      {"/Tile_8_output_0_scale", 0.0003696007188409567f},
      {"/Slice_32_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_44_output_0_scale", 0.0003696007188409567f},
      {"/Add_42_output_0_scale", 0.0007873804424889386f},
      {"/Concat_186_output_0_scale", 0.00015599273319821805f},
      {"/Mul_45_output_0_scale", 2.935774864454288e-05f},
      {"/MatMul_27_output_0_scale", 0.0002783653326332569f},
      {"/MatMul_28_output_0_scale", 0.000291624222882092f},
      {"/Add_45_output_0_scale", 0.00035984389251098037f},
      {"/Slice_37_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_48_output_0_scale", 0.00035984389251098037f},
      {"/ReduceMin_9_output_0_scale", 0.00017802380898501724f},
      {"/Expand_9_output_0_scale", 0.00017802380898501724f},
      {"/Tile_9_output_0_scale", 0.00017802380898501724f},
      {"/Slice_36_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_49_output_0_scale", 0.00017802380898501724f},
      {"/Add_47_output_0_scale", 0.00035984389251098037f},
      {"/Concat_194_output_0_scale", 0.00010930887947324663f},
      {"/Mul_50_output_0_scale", 2.243317612737883e-05f},
      {"/MatMul_30_output_0_scale", 0.00029304061899892986f},
      {"/MatMul_31_output_0_scale", 0.0002636133285705f},
      {"/Add_50_output_0_scale", 0.00036721021751873195f},
      {"/Slice_41_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_53_output_0_scale", 0.0003548910026438534f},
      {"/ReduceMin_10_output_0_scale", 0.00016278795374091715f},
      {"/Expand_10_output_0_scale", 0.00016278795374091715f},
      {"/Tile_10_output_0_scale", 0.00016278795374091715f},
      {"/Slice_40_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_54_output_0_scale", 0.00016278795374091715f},
      {"/Add_52_output_0_scale", 0.0003548910026438534f},
      {"/Concat_202_output_0_scale", 0.00012250816507730633f},
      {"/Mul_55_output_0_scale", 2.5055691367015243e-05f},
      {"/MatMul_33_output_0_scale", 0.0002806176198646426f},
      {"/MatMul_34_output_0_scale", 0.0003001239092554897f},
      {"/Add_55_output_0_scale", 0.0003663137322291732f},
      {"/Slice_45_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_58_output_0_scale", 0.0003663137322291732f},
      {"/ReduceMin_11_output_0_scale", 0.0001697707484709099f},
      {"/Expand_11_output_0_scale", 0.0001697707484709099f},
      {"/Tile_11_output_0_scale", 0.0001697707484709099f},
      {"/Slice_44_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_59_output_0_scale", 0.0001697707484709099f},
      {"/Add_57_output_0_scale", 0.0003663137322291732f},
      {"/Concat_210_output_0_scale", 0.00014686973008792847f},
      {"/Mul_60_output_0_scale", 3.8016234611859545e-05f},
      {"/MatMul_36_output_0_scale", 0.0003973279963247478f},
      {"/MatMul_37_output_0_scale", 0.00044878062908537686f},
      {"/Add_60_output_0_scale", 0.0005884782294742763f},
      {"/Slice_49_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_63_output_0_scale", 0.0005884782294742763f},
      {"/ReduceMin_12_output_0_scale", 0.0002618382277432829f},
      {"/Expand_12_output_0_scale", 0.0002618382277432829f},
      {"/Tile_12_output_0_scale", 0.0002618382277432829f},
      {"/Slice_48_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_64_output_0_scale", 0.0002618382277432829f},
      {"/Add_62_output_0_scale", 0.0005884782294742763f},
      {"/Concat_218_output_0_scale", 0.00020529770699795336f},
      {"/Mul_65_output_0_scale", 2.7692027288139798e-05f},
      {"/MatMul_39_output_0_scale", 0.0003990688710473478f},
      {"/MatMul_40_output_0_scale", 0.000294659985229373f},
      {"/Add_65_output_0_scale", 0.00047001460916362703f},
      {"/Slice_53_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_68_output_0_scale", 0.00047001460916362703f},
      {"/ReduceMin_13_output_0_scale", 0.00025355833349749446f},
      {"/Expand_13_output_0_scale", 0.00025355833349749446f},
      {"/Tile_13_output_0_scale", 0.00025355833349749446f},
      {"/Slice_52_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_69_output_0_scale", 0.00025355833349749446f},
      {"/Add_67_output_0_scale", 0.00047001460916362703f},
      {"/Concat_226_output_0_scale", 0.00013713071530219167f},
      {"/Mul_70_output_0_scale", 2.3133450667955913e-05f},
      {"/MatMul_42_output_0_scale", 0.00027545166085474193f},
      {"/MatMul_43_output_0_scale", 0.00024557820870541036f},
      {"/Add_70_output_0_scale", 0.0004187080485280603f},
      {"/Slice_57_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_73_output_0_scale", 0.00040808069752529263f},
      {"/ReduceMin_14_output_0_scale", 0.00020171620417386293f},
      {"/Expand_14_output_0_scale", 0.00020171620417386293f},
      {"/Tile_14_output_0_scale", 0.00020171620417386293f},
      {"/Slice_56_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_74_output_0_scale", 0.00020171620417386293f},
      {"/Add_72_output_0_scale", 0.00040808069752529263f},
      {"/Concat_234_output_0_scale", 0.00014357810141518712f},
      {"/Mul_75_output_0_scale", 1.9944669475080445e-05f},
      {"/MatMul_45_output_0_scale", 0.0003169697301927954f},
      {"/MatMul_46_output_0_scale", 0.00022894710127729923f},
      {"/Add_75_output_0_scale", 0.0003733447811100632f},
      {"/Slice_61_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_78_output_0_scale", 0.0003733447811100632f},
      {"/ReduceMin_15_output_0_scale", 0.00021414588263724f},
      {"/Expand_15_output_0_scale", 0.00021414588263724f},
      {"/Tile_15_output_0_scale", 0.00021414588263724f},
      {"/Slice_60_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_79_output_0_scale", 0.00021414588263724f},
      {"/Add_77_output_0_scale", 0.0003733447811100632f},
      {"/Concat_242_output_0_scale", 0.00015407624596264213f},
      {"/Mul_80_output_0_scale", 2.1508438294404186e-05f},
      {"/MatMul_48_output_0_scale", 0.00028399063739925623f},
      {"/MatMul_49_output_0_scale", 0.00023845327086746693f},
      {"/Add_80_output_0_scale", 0.00036361542879603803f},
      {"/Slice_65_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_83_output_0_scale", 0.00036361542879603803f},
      {"/ReduceMin_16_output_0_scale", 0.0002237686567241326f},
      {"/Expand_16_output_0_scale", 0.0002237686567241326f},
      {"/Tile_16_output_0_scale", 0.0002237686567241326f},
      {"/Slice_64_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_84_output_0_scale", 0.0002237686567241326f},
      {"/Add_82_output_0_scale", 0.00036361542879603803f},
      {"/Concat_250_output_0_scale", 0.0001593720371602103f},
      {"/Mul_85_output_0_scale", 2.163218414352741e-05f},
      {"/MatMul_51_output_0_scale", 0.00030275460449047387f},
      {"/MatMul_52_output_0_scale", 0.00022170942975208163f},
      {"/Add_85_output_0_scale", 0.00039568060310557485f},
      {"/Slice_69_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_88_output_0_scale", 0.00039568060310557485f},
      {"/ReduceMin_17_output_0_scale", 0.00026992958737537265f},
      {"/Expand_17_output_0_scale", 0.00026992958737537265f},
      {"/Tile_17_output_0_scale", 0.00026992958737537265f},
      {"/Slice_68_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_89_output_0_scale", 0.00026992958737537265f},
      {"/Add_87_output_0_scale", 0.00039568060310557485f},
      {"/Concat_258_output_0_scale", 0.00016141663945745677f},
      {"/Mul_90_output_0_scale", 2.1820380425197072e-05f},
      {"/MatMul_54_output_0_scale", 0.0003278267686255276f},
      {"/MatMul_55_output_0_scale", 0.00022389051446225494f},
      {"/Add_90_output_0_scale", 0.0004276778781786561f},
      {"/Slice_73_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_93_output_0_scale", 0.0004276778781786561f},
      {"/ReduceMin_18_output_0_scale", 0.00029052409809082747f},
      {"/Expand_18_output_0_scale", 0.00029052409809082747f},
      {"/Tile_18_output_0_scale", 0.00029052409809082747f},
      {"/Slice_72_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_94_output_0_scale", 0.00029052409809082747f},
      {"/Add_92_output_0_scale", 0.0004276778781786561f},
      {"/Concat_266_output_0_scale", 0.0002001774701057002f},
      {"/Mul_95_output_0_scale", 3.484821354504675e-05f},
      {"/MatMul_57_output_0_scale", 0.0004294230602681637f},
      {"/MatMul_58_output_0_scale", 0.00035620472044683993f},
      {"/Add_95_output_0_scale", 0.0005414583720266819f},
      {"/Slice_77_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_98_output_0_scale", 0.0005414583720266819f},
      {"/ReduceMin_19_output_0_scale", 0.0003105315554421395f},
      {"/Expand_19_output_0_scale", 0.0003105315554421395f},
      {"/Tile_19_output_0_scale", 0.0003105315554421395f},
      {"/Slice_76_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_99_output_0_scale", 0.0003105315554421395f},
      {"/Add_97_output_0_scale", 0.0005414583720266819f},
      {"/Concat_274_output_0_scale", 0.00014745416410733014f},
      {"/Mul_100_output_0_scale", 2.2222786355996504e-05f},
      {"/MatMul_60_output_0_scale", 0.0002937836106866598f},
      {"/MatMul_61_output_0_scale", 0.00021547595679294318f},
      {"/Add_100_output_0_scale", 0.0003655809268821031f},
      {"/Slice_81_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_103_output_0_scale", 0.0003655809268821031f},
      {"/ReduceMin_20_output_0_scale", 0.0002440168464090675f},
      {"/Expand_20_output_0_scale", 0.0002440168464090675f},
      {"/Tile_20_output_0_scale", 0.0002440168464090675f},
      {"/Slice_80_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_104_output_0_scale", 0.0002440168464090675f},
      {"/Add_102_output_0_scale", 0.0003655809268821031f},
      {"/Concat_282_output_0_scale", 0.00018875901878345758f},
      {"/Mul_105_output_0_scale", 2.469232458679471e-05f},
      {"/MatMul_63_output_0_scale", 0.0003721622342709452f},
      {"/MatMul_64_output_0_scale", 0.00022797904966864735f},
      {"/Add_105_output_0_scale", 0.00045641310862265527f},
      {"/Slice_85_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_108_output_0_scale", 0.00045641310862265527f},
      {"/ReduceMin_21_output_0_scale", 0.00034466441138647497f},
      {"/Expand_21_output_0_scale", 0.00034466441138647497f},
      {"/Tile_21_output_0_scale", 0.00034466441138647497f},
      {"/Slice_84_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_109_output_0_scale", 0.00034466441138647497f},
      {"/Add_107_output_0_scale", 0.00045641310862265527f},
      {"/Concat_290_output_0_scale", 0.00018305757839698344f},
      {"/Mul_110_output_0_scale", 2.6395786335342564e-05f},
      {"/MatMul_66_output_0_scale", 0.00038010653224773705f},
      {"/MatMul_67_output_0_scale", 0.00023804335796739906f},
      {"/Add_110_output_0_scale", 0.00046271338942460716f},
      {"/Slice_89_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_113_output_0_scale", 0.00046271338942460716f},
      {"/ReduceMin_22_output_0_scale", 0.0003383342700544745f},
      {"/Expand_22_output_0_scale", 0.0003383342700544745f},
      {"/Tile_22_output_0_scale", 0.0003383342700544745f},
      {"/Slice_88_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_114_output_0_scale", 0.0003383342700544745f},
      {"/Add_112_output_0_scale", 0.00046271338942460716f},
      {"/Concat_298_output_0_scale", 0.00017657957505434752f},
      {"/Mul_115_output_0_scale", 2.4372899133595638e-05f},
      {"/MatMul_69_output_0_scale", 0.0003723691334016621f},
      {"/MatMul_70_output_0_scale", 0.0002461915137246251f},
      {"/Add_115_output_0_scale", 0.0004571039753500372f},
      {"/Slice_93_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_118_output_0_scale", 0.0004571039753500372f},
      {"/ReduceMin_23_output_0_scale", 0.00034125029924325645f},
      {"/Expand_23_output_0_scale", 0.00034125029924325645f},
      {"/Tile_23_output_0_scale", 0.00034125029924325645f},
      {"/Slice_92_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_119_output_0_scale", 0.00034125029924325645f},
      {"/Add_117_output_0_scale", 0.0004571039753500372f},
      {"/Concat_306_output_0_scale", 0.0002360977669013664f},
      {"/Mul_120_output_0_scale", 3.032855238416232e-05f},
      {"/MatMul_72_output_0_scale", 0.00046915089478716254f},
      {"/MatMul_73_output_0_scale", 0.00027626368682831526f},
      {"/Add_120_output_0_scale", 0.0005828705034218729f},
      {"/Slice_97_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_123_output_0_scale", 0.0005828705034218729f},
      {"/ReduceMin_24_output_0_scale", 0.00044110228191129863f},
      {"/Expand_24_output_0_scale", 0.00044110228191129863f},
      {"/Tile_24_output_0_scale", 0.00044110228191129863f},
      {"/Slice_96_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_124_output_0_scale", 0.00044110228191129863f},
      {"/Add_122_output_0_scale", 0.0005828705034218729f},
      {"/Concat_314_output_0_scale", 0.00023472582688555121f},
      {"/Mul_125_output_0_scale", 2.7340289307176135e-05f},
      {"/MatMul_75_output_0_scale", 0.00042846554424613714f},
      {"/MatMul_76_output_0_scale", 0.0002434732741676271f},
      {"/Add_125_output_0_scale", 0.0005324677331373096f},
      {"/Slice_101_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_128_output_0_scale", 0.0005324677331373096f},
      {"/ReduceMin_25_output_0_scale", 0.00036098252166993916f},
      {"/Expand_25_output_0_scale", 0.00036098252166993916f},
      {"/Tile_25_output_0_scale", 0.00036098252166993916f},
      {"/Slice_100_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_129_output_0_scale", 0.00036098252166993916f},
      {"/Add_127_output_0_scale", 0.0005324677331373096f},
      {"/Concat_322_output_0_scale", 0.00021961952734272927f},
      {"/Mul_130_output_0_scale", 3.183111039106734e-05f},
      {"/MatMul_78_output_0_scale", 0.00042310921708121896f},
      {"/MatMul_79_output_0_scale", 0.00025201914831995964f},
      {"/Add_130_output_0_scale", 0.0005276371375657618f},
      {"/Slice_105_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_133_output_0_scale", 0.0005276371375657618f},
      {"/ReduceMin_26_output_0_scale", 0.00038994534406811f},
      {"/Expand_26_output_0_scale", 0.00038994534406811f},
      {"/Tile_26_output_0_scale", 0.00038994534406811f},
      {"/Slice_104_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_134_output_0_scale", 0.00038994534406811f},
      {"/Add_132_output_0_scale", 0.0005276371375657618f},
      {"/Concat_330_output_0_scale", 0.00016913408762775362f},
      {"/Mul_135_output_0_scale", 2.5507404643576592e-05f},
      {"/MatMul_81_output_0_scale", 0.00040568842086941004f},
      {"/MatMul_82_output_0_scale", 0.00027708703419193625f},
      {"/Add_135_output_0_scale", 0.000521214387845248f},
      {"/Slice_109_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_138_output_0_scale", 0.000521214387845248f},
      {"/ReduceMin_27_output_0_scale", 0.0003497916623018682f},
      {"/Expand_27_output_0_scale", 0.0003497916623018682f},
      {"/Tile_27_output_0_scale", 0.0003497916623018682f},
      {"/Slice_108_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_139_output_0_scale", 0.0003497916623018682f},
      {"/Add_137_output_0_scale", 0.000521214387845248f},
      {"/Concat_338_output_0_scale", 0.00015404335863422602f},
      {"/Mul_140_output_0_scale", 2.3934813725645654e-05f},
      {"/MatMul_84_output_0_scale", 0.0004186215519439429f},
      {"/MatMul_85_output_0_scale", 0.00026420780341140926f},
      {"/Add_140_output_0_scale", 0.0004958264180459082f},
      {"/Slice_113_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_143_output_0_scale", 0.0004886874230578542f},
      {"/ReduceMin_28_output_0_scale", 0.00035233653034083545f},
      {"/Expand_28_output_0_scale", 0.00035233653034083545f},
      {"/Tile_28_output_0_scale", 0.00035233653034083545f},
      {"/Slice_112_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_144_output_0_scale", 0.00035233653034083545f},
      {"/Add_142_output_0_scale", 0.0004886874230578542f},
      {"/Concat_346_output_0_scale", 0.0001701028668321669f},
      {"/Mul_145_output_0_scale", 2.8449303499655798e-05f},
      {"/MatMul_87_output_0_scale", 0.0003773857606574893f},
      {"/MatMul_88_output_0_scale", 0.0002495612425263971f},
      {"/Add_145_output_0_scale", 0.00043686749995686114f},
      {"/Slice_117_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_148_output_0_scale", 0.00043686749995686114f},
      {"/ReduceMin_29_output_0_scale", 0.0002816251653712243f},
      {"/Expand_29_output_0_scale", 0.0002816251653712243f},
      {"/Tile_29_output_0_scale", 0.0002816251653712243f},
      {"/Slice_116_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_149_output_0_scale", 0.0002816251653712243f},
      {"/Add_147_output_0_scale", 0.00043686749995686114f},
      {"/Concat_354_output_0_scale", 0.00018713991448748857f},
      {"/Mul_150_output_0_scale", 5.339468771126121e-05f},
      {"/MatMul_90_output_0_scale", 0.00043079856550320983f},
      {"/MatMul_91_output_0_scale", 0.0003364064614288509f},
      {"/Add_150_output_0_scale", 0.0006598882609978318f},
      {"/Slice_121_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_153_output_0_scale", 0.0006408260087482631f},
      {"/ReduceMin_30_output_0_scale", 0.00030419384711422026f},
      {"/Expand_30_output_0_scale", 0.00030419384711422026f},
      {"/Tile_30_output_0_scale", 0.00030419384711422026f},
      {"/Slice_120_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_154_output_0_scale", 0.00030419384711422026f},
      {"/Add_152_output_0_scale", 0.0006408260087482631f},
      {"/Concat_362_output_0_scale", 0.00020513652998488396f},
      {"/Mul_155_output_0_scale", 2.7092677555629052e-05f},
      {"/MatMul_93_output_0_scale", 0.0004125100967939943f},
      {"/MatMul_94_output_0_scale", 0.0002750830608420074f},
      {"/Add_155_output_0_scale", 0.0005437650252133608f},
      {"/Slice_125_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_158_output_0_scale", 0.0005437650252133608f},
      {"/ReduceMin_31_output_0_scale", 0.00035365880466997623f},
      {"/Expand_31_output_0_scale", 0.00035365880466997623f},
      {"/Tile_31_output_0_scale", 0.00035365880466997623f},
      {"/Slice_124_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_159_output_0_scale", 0.00035365880466997623f},
      {"/Add_157_output_0_scale", 0.0005437650252133608f},
      {"/Concat_370_output_0_scale", 0.00021271371224429458f},
      {"/Mul_160_output_0_scale", 4.598019222612493e-05f},
      {"/MatMul_96_output_0_scale", 0.0005290820845402777f},
      {"/MatMul_97_output_0_scale", 0.00034505699295550585f},
      {"/Add_160_output_0_scale", 0.0007383169722743332f},
      {"/Slice_129_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_163_output_0_scale", 0.0007383169722743332f},
      {"/ReduceMin_32_output_0_scale", 0.00028465871582739055f},
      {"/Expand_32_output_0_scale", 0.00028465871582739055f},
      {"/Tile_32_output_0_scale", 0.00028465871582739055f},
      {"/Slice_128_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_164_output_0_scale", 0.00028465871582739055f},
      {"/Add_162_output_0_scale", 0.0007383169722743332f},
      {"/Concat_378_output_0_scale", 0.00025845563504844904f},
      {"/Mul_165_output_0_scale", 3.526033106027171e-05f},
      {"/MatMul_99_output_0_scale", 0.00040739361429587007f},
      {"/MatMul_100_output_0_scale", 0.00027854053769260645f},
      {"/Add_165_output_0_scale", 0.0005625546327792108f},
      {"/Slice_133_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_168_output_0_scale", 0.0005625546327792108f},
      {"/ReduceMin_33_output_0_scale", 0.0002677535230759531f},
      {"/Expand_33_output_0_scale", 0.0002677535230759531f},
      {"/Tile_33_output_0_scale", 0.0002677535230759531f},
      {"/Slice_132_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_169_output_0_scale", 0.0002677535230759531f},
      {"/Add_167_output_0_scale", 0.0005625546327792108f},
      {"/Concat_386_output_0_scale", 0.0002565381582826376f},
      {"/Mul_170_output_0_scale", 3.345478035043925e-05f},
      {"/MatMul_102_output_0_scale", 0.00036085298052057624f},
      {"/MatMul_103_output_0_scale", 0.0002558672276791185f},
      {"/Add_170_output_0_scale", 0.00047741163871251047f},
      {"/Slice_137_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_173_output_0_scale", 0.00047741163871251047f},
      {"/ReduceMin_34_output_0_scale", 0.00025433284463360906f},
      {"/Expand_34_output_0_scale", 0.00025433284463360906f},
      {"/Tile_34_output_0_scale", 0.00025433284463360906f},
      {"/Slice_136_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_174_output_0_scale", 0.00025433284463360906f},
      {"/Add_172_output_0_scale", 0.00047741163871251047f},
      {"/Concat_394_output_0_scale", 0.00024195745936594903f},
      {"/Mul_175_output_0_scale", 3.328558523207903e-05f},
      {"/MatMul_105_output_0_scale", 0.00035910855513066053f},
      {"/MatMul_106_output_0_scale", 0.0002520213311072439f},
      {"/Add_175_output_0_scale", 0.000487500277813524f},
      {"/Slice_141_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_178_output_0_scale", 0.000487500277813524f},
      {"/ReduceMin_35_output_0_scale", 0.0002569222415331751f},
      {"/Expand_35_output_0_scale", 0.0002569222415331751f},
      {"/Tile_35_output_0_scale", 0.0002569222415331751f},
      {"/Slice_140_output_0_scale", 1.5259021893143654e-05f},
      {"/Mul_179_output_0_scale", 0.0002569222415331751f},
      {"/Add_177_output_0_scale", 0.000487500277813524f},
      {"/Add_179_output_0_scale", 0.03730785846710205f},
      {"encoder.after_norm.weight_scale", 0.0005806717090308666f},
      {"encoder.after_norm.bias_quantized_scale", 0.000021663618099410087f},
      {"/after_norm/Add_1_output_0_scale", 0.000020904186385450885f},
      {"/lin_enc/fc/Transpose_output_0_scale", 0.000016458583559142426f},
      {"/lin_enc/fc/MatMul_output_0_scale", 0.00008215451089199632f},
      {"joint_network.lin_enc.fc.bias_scale", 0.000009380423762195278f},
      {"/lin_enc/fc/MatMul_output_0_scale", 0.00008215451089199632f},
      {"/lin_enc/fc/Add_output_0_scale", 0.00009135611617239192f},
      {"joint_network.lin_enc.Lnorm.weight_scale", 0.07113970816135406f},
      {"joint_network.lin_enc.Lnorm.bias_quantized_scale",
       0.000006499047231045552f},
      {"hidden_state_scale", 0.0029028654098510742f},
      {"cache_frames_scale", 0.000409241154557094f},
      {"encoder_embedding.global_mean_scale", 0.00023770694679114968f},
      {"encoder_embedding.global_mean_scale", 0.00023770694679114968f},
      {"/encoder_embedding/Sub_output_0_scale", 0.00045756122563034296f},
      {"encoder_embedding.global_invstd_scale", 0.000003978670974902343f},
      {"/encoder_embedding/Mul_output_0_scale", 0.00010564539843471721f},
      {"encoder.embed.conv.0.weight_scale", 0.002004346577450633f},
      {"encoder.embed.conv.0.bias_quantized_scale", 2.117499917630994e-7f},
      {"/conv/conv.1/Relu_output_0_scale", 0.00002364343345107045f},
      {"encoder.embed.conv.2.weight_scale", 0.01033624354749918f},
      {"encoder.embed.conv.2.bias_quantized_scale", 2.443842959110043e-7f},
      {"/conv/conv.3/Relu_output_0_scale", 0.000023254606276168488f},
      {"/out/Transpose_output_0_scale", 0.0000313227174046915f},
      {"/out/MatMul_output_0_scale", 0.00020969474280718714f},
      {"encoder.embed.out.bias_scale", 0.00001832125781220384f},
      {"/out/Add_output_0_scale", 0.0002087922766804695f},
      {"encoder_embedding.global_mean_scale", 0.00023770694679114968f}}}};
// mm wts
inline int64_t rnd_to_even(float float_val) {
  float float_val_frac = float_val - std::floor(float_val);
  int64_t float_val_rnd = (int64_t)std::round(float_val);
  int64_t res = float_val_rnd;
  if ((float_val_frac == 0.5) && ((float_val_rnd % 2) == 1)) {
    res -= 1;
  }
  return res;
}
inline void binary_op_wts_write(const std::vector<uint8_t>& wts_vec,
                                int8_t*& wts_ptr, int8_t*& rtp_ptr) {
  if (rtp_ptr == nullptr) {
    memcpy(wts_ptr, wts_vec.data(), wts_vec.size());
    wts_ptr += wts_vec.size();
  } else {
    memcpy(rtp_ptr, wts_vec.data(), 64);
    memcpy(wts_ptr, wts_vec.data() + 64, wts_vec.size() - 64);
    rtp_ptr += 64;
    wts_ptr += wts_vec.size() - 64;
  }
}

template <typename weight_type>
std::vector<uint8_t> mm_wts_write(weight_type* weights, int64_t c2, int64_t c1,
                                  std::vector<int64_t>& c0,
                                  int64_t gemm_shift_out, int64_t qdq_shift_out,
                                  uint32_t gemm_m, uint32_t gemm_k,
                                  uint32_t gemm_n, uint32_t sv_M = 32,
                                  uint32_t sv_K = 64, uint32_t sv_N = 16) {
  const uint32_t NUM_COLS = 4;
  const uint32_t NUM_ROWS = 4;
  std::vector<uint8_t> result(
      gemm_k * gemm_n * sizeof(weight_type) +
          (sv_N * 2 + 2) * 4 * gemm_k / sv_K * gemm_n / sv_N + 64,
      0);
  uint8_t* result_ptr = result.data();
  uint32_t RTP_cp[16] = {
      sv_M,
      sv_K,
      sv_N,
      (((gemm_m + sv_M - 1) / sv_M) << 16) | ((gemm_k / sv_K) << 8) |
          (gemm_n / sv_N / NUM_COLS / NUM_ROWS),
      0x2000,
      0x4800,
      0x3800,
      static_cast<uint32_t>(gemm_shift_out | (qdq_shift_out << 16)),
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      2};
  std::vector<uint32_t> RTP;
  RTP.insert(RTP.begin(), RTP_cp, RTP_cp + 16);
  memcpy(result_ptr, reinterpret_cast<char*>(RTP.data()),
         RTP.size() * sizeof(int32_t));
  result_ptr += RTP.size() * sizeof(int32_t);

  auto sv_n8_formater = [sv_K, sv_N, gemm_n, gemm_k](weight_type* weights,
                                                     int idx_K, int idx_N) {
    std::vector<weight_type> sv_W(sv_K * sv_N, 0);
    for (int ind_row = 0; ind_row < sv_K; ind_row++) {
      for (int ind_col_group = 0; ind_col_group < sv_N / 8; ind_col_group++) {
        for (int ind_col = 0; ind_col < 8; ind_col++) {
          sv_W[ind_col_group * sv_K * 8 + ind_row * 8 + ind_col] =
              weights[(idx_K * sv_K + ind_row) * gemm_n + idx_N * sv_N +
                      ind_col_group * 8 + ind_col];
        }
      }
    }
    return sv_W;
  };

  for (int n3 = 0; n3 < NUM_COLS; n3++) {
    for (int n2 = 0; n2 < gemm_n / NUM_COLS / NUM_ROWS / sv_N; n2++) {
      for (int idx_K = 0; idx_K < gemm_k / sv_K; idx_K++) {
        for (int n1 = 0; n1 < NUM_ROWS; n1++) {
          int idx_N = n3 * (gemm_n / NUM_COLS / sv_N) + n2 * NUM_ROWS + n1;
          std::vector<weight_type> sv_W = sv_n8_formater(weights, idx_K, idx_N);

          memcpy(result_ptr, reinterpret_cast<char*>(sv_W.data()),
                 sv_K * sv_N * sizeof(weight_type));
          result_ptr += sv_K * sv_N * sizeof(weight_type);

          // qdq_coeff
          std::vector<int32_t> qdq_coeff(2 * sv_N + 2, 0);
          for (int i = 0; i < sv_N; ++i) {
            int64_t c0_num = c0[idx_N * sv_N + i];

            qdq_coeff[2 * i] = ((int32_t*)&c0_num)[0];
            qdq_coeff[2 * i + 1] = ((int32_t*)&c0_num)[1];
          }

          qdq_coeff[2 * sv_N] = c1;
          qdq_coeff[2 * sv_N + 1] = c2;

          memcpy(result_ptr, reinterpret_cast<char*>(qdq_coeff.data()),
                 qdq_coeff.size() * sizeof(int32_t));
          result_ptr += qdq_coeff.size() * sizeof(int32_t);
        }
      }
    }
  }

  return result;
}

inline std::pair<int64_t, int64_t> find_closest_shifted_int16(float float_val) {
  int64_t int16_max = 32767;
  double prev_rel_err = 1e9;
  double curr_float_val = float_val;
  double best_float_val = 0.0;
  int64_t shift_val = 0;
  int64_t best_int = 0;
  int64_t closest_curr_int = 0;
  int64_t best_shift_val = 0;

  // std::cout << "[find_closest_shifted_int16]: float_val " << float_val <<
  // std::endl;
  while (curr_float_val <= int16_max) {
    closest_curr_int = rnd_to_even(curr_float_val);
    double cur_rel_err =
        std::abs(float_val - closest_curr_int / std::pow(2, shift_val)) /
        float_val;
    // std::cout << "[find_closest_shifted_int16]: current float_val " <<
    // closest_curr_int / std::pow(2, shift_val)
    //           << ", closest_curr_int " << closest_curr_int << ", shift_val "
    //           << shift_val
    //           << ", cur_rel_err " << cur_rel_err << std::endl;

    if (cur_rel_err < prev_rel_err) {
      prev_rel_err = cur_rel_err;
      best_float_val = static_cast<double>(closest_curr_int >> shift_val);
      best_shift_val = shift_val;
      best_int = closest_curr_int;
    }

    curr_float_val *= 2;
    shift_val++;
  }

  return {best_int, best_shift_val};
}

template <typename weight_type>
std::vector<uint8_t> fused_matmul_bias_A16W16_generate_wts(
    weight_type* weights, uint16_t* bias, float ifm1_scale,
    int64_t ifm1_zero_point, float ifm2_scale, int64_t ifm2_zero_point,
    float bias_scale, int64_t bias_zero_point, float ofm_scale,
    int64_t ofm_zero_point, uint32_t gemm_m, uint32_t gemm_k, uint32_t gemm_n,
    uint32_t sv_M = 32, uint32_t sv_K = 64, uint32_t sv_N = 16) {

  int64_t matmul_shift;
  if (std::is_same<weight_type, uint8_t>::value) {
    matmul_shift =
        gt_min(gt_max((int32_t)(25 + std::ceil(std::log2(gemm_k)) - 32), 0), 7);
  } else {
    printf("using uint16 version of fused matmul!\n");
    matmul_shift = gt_min(
        gt_max((int32_t)(33 + std::ceil(std::log2(gemm_k)) - 32), 0), 15);
  }
  float c2_fp, c4_fp;
  c2_fp = (float)((ifm1_scale * ifm2_scale) / ofm_scale);
  c4_fp = (float)(bias_scale / ofm_scale);

  auto res0 = find_closest_shifted_int16(c2_fp);
  int64_t c2_coeff_prime = res0.first;
  int64_t shft_c2 = res0.second;

  auto res1 = find_closest_shifted_int16(c4_fp);
  int64_t c4_coeff_prime = res1.first;
  int64_t shft_c4 = res1.second;

  if (shft_c2 != shft_c4) {
    int64_t diff_shft_c2_c4 = shft_c2 - shft_c4;
    int64_t abs_diff_shft_c2_c4 = std::abs(diff_shft_c2_c4);
    if (diff_shft_c2_c4 > 0) {
      c4_coeff_prime = c4_coeff_prime << abs_diff_shft_c2_c4;
    } else if (diff_shft_c2_c4 < 0) {
      c4_coeff_prime = c4_coeff_prime >> abs_diff_shft_c2_c4;
    } else {
      c4_coeff_prime = c4_coeff_prime;
    }
  }

  std::vector<int64_t> sum0(gemm_n, 0);
  std::vector<int64_t> c1_coeff(gemm_n, 0);
  std::vector<int64_t> bias_min_zp(gemm_n, 0);
  int64_t offset;
  for (int i = 0; i < gemm_n; i++) {
    for (int j = 0; j < gemm_k; j++) {
      offset = j * gemm_n + i;
      sum0[i] += (int64_t) * (weights + offset);
    }
    bias_min_zp[i] = bias[i] - bias_zero_point;
    c1_coeff[i] = (-ifm1_zero_point) * c2_coeff_prime * sum0[i] +
                  (ofm_zero_point << shft_c2) + bias_min_zp[i] * c4_coeff_prime;
  }

  int64_t c3_coeff_offset = -ifm1_zero_point * gemm_k;
  int64_t c3_coeff_scale = -c2_coeff_prime * ifm2_zero_point;

  int64_t c3_coeff_scale_shift = 0;
  if (std::abs(c3_coeff_scale) > 2147483647) {
    c3_coeff_scale_shift = std::ceil(std::log2(std::abs(c3_coeff_scale))) - 31;
  }
  c3_coeff_scale = c3_coeff_scale >> c3_coeff_scale_shift;

  int64_t C2 = c2_coeff_prime << matmul_shift;
  int64_t C1 = c3_coeff_scale;
  std::vector<int64_t> C0(gemm_n, 0);
  for (int j = 0; j < gemm_n; j++) {
    int64_t cst0 = c3_coeff_offset << c3_coeff_scale_shift;
    int64_t cst1 = c3_coeff_scale * cst0;
    C0[j] = cst1 + c1_coeff[j];
  }
  // printf("matmul_shift: %d, qdq_shift %d\n", matmul_shift, shft_c2);
  std::vector<uint8_t> res_uint8 =
      mm_wts_write<weight_type>(weights, C2, C1, C0, matmul_shift, shft_c2,
                                gemm_m, gemm_k, gemm_n, sv_M, sv_K, sv_N);
  const size_t mm_wts_sz = res_uint8.size();
  const size_t mm_bias_sz = gemm_m * gemm_n * sizeof(uint16_t);
  // res_uint8.resize(mm_wts_sz + mm_bias_sz);
  // memcpy(res_uint8.data() + mm_wts_sz, bias, mm_bias_sz);
  // for(int i = 0; i < 128; ++i) {
  //   std::cout << std::setw(8) << std::hex<<std::setfill('0') <<
  //   *((uint32_t*)res_uint8.data() + i) << std::endl;
  // }
  return res_uint8;
}

inline std::vector<uint8_t> matmul_A16W16_generate_wts(
    uint16_t* weights, float ifm1_scale, int64_t ifm1_zero_point,
    float ifm2_scale, int64_t ifm2_zero_point, float ofm_scale,
    int64_t ofm_zero_point, uint32_t gemm_m, uint32_t gemm_k, uint32_t gemm_n,
    int32_t sv_M = 32, int32_t sv_K = 64, int32_t sv_N = 16) {
// std::cout << "[qdq_matmul_uint16_uint16]: gemm_k " << gemm_k << std::endl;
#define gt_min(a, b) ((a < b) ? a : b)
#define gt_max(a, b) ((a < b) ? b : a)

  int64_t matmul_shift = 0;
  // if (A_DTYPE == "uint8" && W_DTYPE == "uint16") {
  //     matmul_shift = std::min(std::max((int32_t)(25 +
  //     std::ceil(std::log2(gemm_k)) - 32), 0), 7);
  // } else if (A_DTYPE == "uint16" && W_DTYPE == "uint16") {
  matmul_shift =
      gt_min(gt_max((int32_t)(33 + std::ceil(std::log2(gemm_k)) - 32), 0), 15);
  // } else if (A_DTYPE == "uint16" && W_DTYPE == "uint8") {
  //     matmul_shift = std::min(std::max((int32_t)(25 +
  //     std::ceil(std::log2(gemm_k)) - 32), 0), 7);
  // }

  float c2_fp;
  c2_fp = (float)((ifm1_scale * ifm2_scale) / ofm_scale);
  auto res0 = find_closest_shifted_int16(c2_fp);
  int64_t c2 = res0.first;
  int64_t shft_c2 = res0.second;
  // std::cout << "[qdq_matmul_uint16_uint16]: C2 " << c2 << " C2_sft " <<
  // shft_c2 << std::endl;

  int64_t c1 = -c2 * ifm2_zero_point;

  int64_t c1_shift = 0;
  if (std::abs(c1) > 2147483647) {
    c1_shift = std::ceil(std::log2(std::abs(c1))) - 31;
  }
  // std::cout << "[qdq_matmul_uint16_uint16]: C1 " << c1 << " C1_sft " <<
  // c1_shift << std::endl;

  std::vector<int64_t> sum0(gemm_n, 0);
  std::vector<int64_t> c0(gemm_n, 0);
  int64_t offset;
  for (int i = 0; i < gemm_n; i++) {
    for (int j = 0; j < gemm_k; j++) {
      offset = j * gemm_n + i;
      sum0[i] += *(weights + offset);
    }
  }
  int64_t cst0 = ofm_zero_point << shft_c2;
  int64_t cst1 = gemm_k * ifm2_zero_point;
  for (int j = 0; j < gemm_n; j++) {
    c0[j] = c2 * ifm1_zero_point * (cst1 - sum0[j]) + cst0;
  }

  // std::cout << "c1: " << c1 << " c2: " << c2
  //           << " gemm_shift_out: " << matmul_shift
  //           << " qdq_shift_out: " << shft_c2 << std::endl;

  std::vector<uint8_t> res_uint8 =
      mm_wts_write(weights, c2 << matmul_shift, c1, c0, matmul_shift, shft_c2,
                   gemm_m, gemm_k, gemm_n, sv_M, sv_K, sv_N);

  return res_uint8;
}

// add wts
struct AddQDQParams {
  int ifm1_shift;
  int ifm2_shift;
  int ifm1_coeff;
  int ifm2_coeff;
  int zero_point_coeff;
  int zero_point_shift;
  int ofm_shift;
};
inline AddQDQParams add_calc_qdq_params(double ifm1_scale, int ifm1_zero_point,
                                        double ifm2_scale, int ifm2_zero_point,
                                        double ofm_scale, int ofm_zero_point) {
  AddQDQParams params;
  int64_t ifm1_shift = std::floor(-std::log2(ifm1_scale / ofm_scale) + 30);
  int64_t ifm2_shift = std::floor(-std::log2(ifm2_scale / ofm_scale) + 30);

  double signed_zp = ofm_zero_point - ifm1_scale * ifm1_zero_point / ofm_scale -
                     ifm2_scale * ifm2_zero_point / ofm_scale;
  int64_t zero_point_shift = std::floor(-std::log2(std::abs(signed_zp)) + 30);
  int64_t ofm_shift = gt_max(gt_max(ifm1_shift, ifm2_shift), zero_point_shift);

  params.ifm1_coeff =
      static_cast<int>(ifm1_scale / ofm_scale * std::pow(2, ifm1_shift));
  params.ifm2_coeff =
      static_cast<int>(ifm2_scale / ofm_scale * std::pow(2, ifm2_shift));
  params.zero_point_coeff =
      static_cast<int>(signed_zp * std::pow(2, zero_point_shift));
  params.ofm_shift = static_cast<int>(ofm_shift);
  params.ifm1_shift = static_cast<int>(ofm_shift - ifm1_shift);
  params.ifm2_shift = static_cast<int>(ofm_shift - ifm2_shift);
  params.zero_point_shift = static_cast<int>(ofm_shift - zero_point_shift);
  // std::cout << "params.ifm1_coeff " << params.ifm1_coeff
  //           << " params.ifm2_coeff " << params.ifm2_coeff
  //           << " params.zero_point_coeff " << params.zero_point_coeff
  //           << " params.ofm_shift " << params.ofm_shift
  //           << "params.ifm1_shift "
  //           << params.ifm1_shift << " params.ifm2_shift " <<
  //           params.ifm2_shift
  //           << " params.zero_point_shift " << params.zero_point_shift
  //           << std::endl;
  return params;
}

inline std::vector<uint8_t> add_get_RTP(int tensor_size,
                                        const AddQDQParams& params) {
  std::vector<uint8_t> RTP(64, 0);
  RTP[0] = tensor_size & 0xFF;
  RTP[1] = (tensor_size >> 8) & 0xFF;
  RTP[2] = (tensor_size >> 16) & 0xFF;
  RTP[3] = (tensor_size >> 24) & 0xFF;
  RTP[4] = params.ifm1_coeff & 0xFF;
  RTP[5] = (params.ifm1_coeff >> 8) & 0xFF;
  RTP[6] = (params.ifm1_coeff >> 16) & 0xFF;
  RTP[7] = (params.ifm1_coeff >> 24) & 0xFF;
  RTP[8] = params.ifm2_coeff & 0xFF;
  RTP[9] = (params.ifm2_coeff >> 8) & 0xFF;
  RTP[10] = (params.ifm2_coeff >> 16) & 0xFF;
  RTP[11] = (params.ifm2_coeff >> 24) & 0xFF;
  RTP[12] = params.zero_point_coeff & 0xFF;
  RTP[13] = (params.zero_point_coeff >> 8) & 0xFF;
  RTP[14] = (params.zero_point_coeff >> 16) & 0xFF;
  RTP[15] = (params.zero_point_coeff >> 24) & 0xFF;
  RTP[16] = params.ofm_shift;
  RTP[17] = params.ifm1_shift;
  RTP[18] = params.ifm2_shift;
  RTP[19] = params.zero_point_shift;
  RTP[20] = (tensor_size / (4096 * 8)) & 0xFF;
  RTP[21] = ((tensor_size / (4096 * 8)) >> 8) & 0xFF;
  RTP[22] = ((tensor_size / (4096 * 8)) >> 16) & 0xFF;
  RTP[23] = ((tensor_size / (4096 * 8)) >> 24) & 0xFF;

  return RTP;
}

inline std::vector<uint8_t>
add_generate_wts(uint16_t* wts_ptr, size_t output_size, size_t wts_size,
                 float ifm1_scale, int ifm1_zero_point, float ifm2_scale,
                 int ifm2_zero_point, float ofm_scale, int ofm_zero_point) {
  std::vector<uint16_t> vector_wts(wts_ptr, wts_ptr + wts_size);

  AddQDQParams params =
      add_calc_qdq_params(ifm1_scale, ifm1_zero_point, ifm2_scale,
                          ifm2_zero_point, ofm_scale, ofm_zero_point);

  int tensor_size = output_size;
  std::vector<uint8_t> RTP = add_get_RTP(tensor_size, params);
  std::vector<uint8_t> result;
  result.resize(64 + output_size * 2);
  uint8_t* result_ptr = result.data();
  // result.reserve(RTP.size() + vector_wts.size() * sizeof(uint16_t));

  // result.insert(result.end(), RTP.begin(), RTP.end());
  memcpy(result_ptr, RTP.data(), 64);
  result_ptr += 64;

  for (int i = 0; i < int(tensor_size / wts_size); i++) {
    // result.insert(result.end(), vector_wts.begin(), vector_wts.end());
    memcpy(result_ptr, wts_ptr, wts_size * 2);
    result_ptr += wts_size * 2;
  }
  return result;
}
template <typename zp_type>
inline void get_s_zp(
    const std::string& name_s, float& op_scale, const std::string& name_zp,
    zp_type& op_zp,
    const std::unordered_map<std::string, flexmlrt::client::ErtIoTypeNew>& wts_,
    const std::string model_version = "GT_v1.2") {
  if (wts_.find(name_s) == wts_.end()) {
    // std::cout << name_s << " not found for" << model_version << std::endl;
    op_scale = scale_maps.at(model_version).at(name_s);
  } else {
    op_scale = *((float*)(wts_.at(name_s).data));
  }
  if (wts_.find(name_zp) == wts_.end()) {
    // std::cout << name_zp << " not found " << std::endl;
    op_zp = (zp_type)0;
  } else {
    op_zp = *((zp_type*)(wts_.at(name_zp).data));
  }
};

inline size_t GT_MMB_WTS_convert_raw_ptr(
    const std::unordered_map<std::string, flexmlrt::client::ErtIoTypeNew>& wts_,
    int8_t*& wts_ptr, int8_t*& rtp_ptr, int8_t* mmb_w_ptr,
    const std::string& mmb_i_scale_name, const std::string& mmb_i_zp_name,
    const std::string& mmb_w_scale_name, const std::string& mmb_w_zp_name,
    const std::string& mmb_b_scale_name, const std::string& mmb_b_zp_name,
    const std::string& mmb_o_scale_name, const std::string& mmb_o_zp_name,
    const std::string& bias_name, uint32_t gemm_m, uint32_t gemm_k,
    uint32_t gemm_n, uint32_t bias_broadcast_num, uint32_t sv_M, uint32_t sv_K,
    uint32_t sv_N, const std::string model_version) {
  using weight_type = uint8_t;
  float mmb_i_scale = 1.0;
  float mmb_w_scale = 1.0;
  float mmb_b_scale = 1.0;
  float mmb_o_scale = 1.0;
  uint16_t mmb_i_zp = 0;
  weight_type mmb_w_zp = 0;
  uint16_t mmb_b_zp = 0;
  uint16_t mmb_o_zp = 0;
  get_s_zp(mmb_i_scale_name, mmb_i_scale, mmb_i_zp_name, mmb_i_zp, wts_,
           model_version);
  get_s_zp<weight_type>(mmb_w_scale_name, mmb_w_scale, mmb_w_zp_name, mmb_w_zp,
                        wts_, model_version);
  get_s_zp(mmb_b_scale_name, mmb_b_scale, mmb_b_zp_name, mmb_b_zp, wts_,
           model_version);
  get_s_zp(mmb_o_scale_name, mmb_o_scale, mmb_o_zp_name, mmb_o_zp, wts_,
           model_version);
  // broadcasting bias
  uint16_t* bias_ptr = (uint16_t*)(wts_.at(bias_name).data);
  auto bias_shape = wts_.at(bias_name).shape;
  size_t bias_wts_size = std::accumulate(bias_shape.begin(), bias_shape.end(),
                                         1, std::multiplies<size_t>());
  size_t bias_broadcasted_size = bias_wts_size * bias_broadcast_num;
  std::vector<uint16_t> bias_broadcasted(bias_broadcasted_size);
  for (int i = 0; i < bias_broadcast_num; ++i) {
    std::memcpy(bias_broadcasted.data() + i * bias_wts_size, bias_ptr,
                bias_wts_size * sizeof(uint16_t));
  }
  // use ptr directly
  std::vector<uint8_t> mmb_wts = fused_matmul_bias_A16W16_generate_wts(
      (uint8_t*)mmb_w_ptr, bias_broadcasted.data(), mmb_i_scale,
      (int64_t)mmb_i_zp, mmb_w_scale, (int64_t)mmb_w_zp, mmb_b_scale,
      (int64_t)mmb_b_zp, mmb_o_scale, (int64_t)mmb_o_zp, gemm_m, gemm_k, gemm_n,
      sv_M, sv_K, sv_N);
  binary_op_wts_write(mmb_wts, wts_ptr, rtp_ptr);
  return mmb_wts.size();
}

inline size_t GT_MMB_WTS_convert_ptr(
    const std::unordered_map<std::string, flexmlrt::client::ErtIoTypeNew>& wts_,
    int8_t*& wts_ptr, int8_t*& rtp_ptr, const std::string& mmb_i_scale_name,
    const std::string& mmb_i_zp_name, const std::string& mmb_w_scale_name,
    const std::string& mmb_w_zp_name, const std::string& mmb_b_scale_name,
    const std::string& mmb_b_zp_name, const std::string& mmb_o_scale_name,
    const std::string& mmb_o_zp_name, const std::string& mmb_w_name,
    const std::string& bias_name, uint32_t gemm_m, uint32_t gemm_k,
    uint32_t gemm_n, uint32_t bias_broadcast_num, uint32_t sv_M, uint32_t sv_K,
    uint32_t sv_N, bool pad_w, const std::string model_version) {
  using weight_type = uint8_t;
  float mmb_i_scale = 1.0;
  float mmb_w_scale = 1.0;
  float mmb_b_scale = 1.0;
  float mmb_o_scale = 1.0;
  uint16_t mmb_i_zp = 0;
  weight_type mmb_w_zp = 0;
  uint16_t mmb_b_zp = 0;
  uint16_t mmb_o_zp = 0;
  get_s_zp(mmb_i_scale_name, mmb_i_scale, mmb_i_zp_name, mmb_i_zp, wts_,
           model_version);
  get_s_zp<weight_type>(mmb_w_scale_name, mmb_w_scale, mmb_w_zp_name, mmb_w_zp,
                        wts_, model_version);
  get_s_zp(mmb_b_scale_name, mmb_b_scale, mmb_b_zp_name, mmb_b_zp, wts_,
           model_version);
  get_s_zp(mmb_o_scale_name, mmb_o_scale, mmb_o_zp_name, mmb_o_zp, wts_,
           model_version);
  // broadcasting bias
  uint16_t* bias_ptr = (uint16_t*)(wts_.at(bias_name).data);
  auto bias_shape = wts_.at(bias_name).shape;
  size_t bias_wts_size = std::accumulate(bias_shape.begin(), bias_shape.end(),
                                         1, std::multiplies<size_t>());
  size_t bias_broadcasted_size = bias_wts_size * bias_broadcast_num;
  std::vector<uint16_t> bias_broadcasted(bias_broadcasted_size);
  for (int i = 0; i < bias_broadcast_num; ++i) {
    std::memcpy(bias_broadcasted.data() + i * bias_wts_size, bias_ptr,
                bias_wts_size * sizeof(uint16_t));
  }
  // pad weights if necessary
  weight_type* w_ptr_orig = (weight_type*)(wts_.at(mmb_w_name).data);
  weight_type* w_ptr_padded = nullptr;
  std::vector<weight_type> w_padded;
  if (pad_w) {
    // currently only used for mm in linear_out, int8 case
    const size_t dim1 = 512;
    const size_t dim2 = 19;
    const size_t dim3 = 512;
    const size_t new_dim2 = 20;
    w_padded.resize(dim1 * new_dim2 * dim3, mmb_w_zp);
    for (int i = 0; i < dim1; ++i) {
      for (int j = 0; j < dim2; ++j) {
        // dim3 elem (of size int8) each time
        std::memcpy(w_padded.data() + (i * new_dim2 + j) * dim3,
                    w_ptr_orig + (i * dim2 + j) * dim3, dim3);
      }
    }
    w_ptr_padded = w_padded.data();
  } else {
    w_ptr_padded = w_ptr_orig;
  }
  std::vector<uint8_t> mmb_wts = fused_matmul_bias_A16W16_generate_wts(
      w_ptr_padded, bias_broadcasted.data(), mmb_i_scale, (int64_t)mmb_i_zp,
      mmb_w_scale, (int64_t)mmb_w_zp, mmb_b_scale, (int64_t)mmb_b_zp,
      mmb_o_scale, (int64_t)mmb_o_zp, gemm_m, gemm_k, gemm_n, sv_M, sv_K, sv_N);
  binary_op_wts_write(mmb_wts, wts_ptr, rtp_ptr);
  return mmb_wts.size();
}

// helper function for InitGtWeight
inline size_t GT_MM_WTS_convert(
    const std::unordered_map<std::string, flexmlrt::client::ErtIoTypeNew>& wts_,
    int8_t*& wts_ptr, int8_t*& rtp_ptr, const std::string& mm_i_scale_name,
    const std::string& mm_i_zp_name, const std::string& mm_w_scale_name,
    const std::string& mm_w_zp_name, const std::string& mm_o_scale_name,
    const std::string& mm_o_zp_name, const std::string& mm_w_name,
    uint32_t gemm_m) {
  // MatMul
  float mm_i_scale = 1.0;
  float mm_w_scale = 1.0;
  float mm_o_scale = 1.0;
  uint16_t mm_i_zp = 0;
  uint16_t mm_w_zp = 0;
  uint16_t mm_o_zp = 0;
  get_s_zp(mm_i_scale_name, mm_i_scale, mm_i_zp_name, mm_i_zp, wts_);
  get_s_zp(mm_w_scale_name, mm_w_scale, mm_w_zp_name, mm_w_zp, wts_);
  get_s_zp(mm_o_scale_name, mm_o_scale, mm_o_zp_name, mm_o_zp, wts_);
  // if (wts_.find(mm_w_name) == wts_.end()) {
  //    std::cout << mm_w_name <<  " not found "<<std::endl;
  //}
  uint16_t* mm_w_ptr = (uint16_t*)(wts_.at(mm_w_name).data);
  auto mm_w_shape = wts_.at(mm_w_name).shape;

  std::vector<uint8_t> mm_wts = matmul_A16W16_generate_wts(
      mm_w_ptr, mm_i_scale, mm_i_zp, mm_w_scale, mm_w_zp, mm_o_scale, mm_o_zp,
      gemm_m, mm_w_shape.at(mm_w_shape.size() - 2),
      mm_w_shape.at(mm_w_shape.size() - 1));

  binary_op_wts_write(mm_wts, wts_ptr, rtp_ptr);
  return mm_wts.size();
}

// helper function for InitGtWeight
inline size_t GT_MM_WTS_convert_ptr(
    const std::unordered_map<std::string, flexmlrt::client::ErtIoTypeNew>& wts_,
    uint16_t* wts_ptr_pad, int32_t gemm_m, int32_t gemm_k, int32_t gemm_n,
    int8_t*& wts_ptr, int8_t*& rtp_ptr, const std::string& mm_i_scale_name,
    const std::string& mm_i_zp_name, const std::string& mm_w_scale_name,
    const std::string& mm_w_zp_name, const std::string& mm_o_scale_name,
    const std::string& mm_o_zp_name, const std::string& mm_w_name) {
  // MatMul
  float mm_i_scale = 1.0;
  float mm_w_scale = 1.0;
  float mm_o_scale = 1.0;
  uint16_t mm_i_zp = 0;
  uint16_t mm_w_zp = 0;
  uint16_t mm_o_zp = 0;
  get_s_zp(mm_i_scale_name, mm_i_scale, mm_i_zp_name, mm_i_zp, wts_);
  get_s_zp(mm_w_scale_name, mm_w_scale, mm_w_zp_name, mm_w_zp, wts_);
  get_s_zp(mm_o_scale_name, mm_o_scale, mm_o_zp_name, mm_o_zp, wts_);
  std::vector<uint8_t> mm_wts = matmul_A16W16_generate_wts(
      wts_ptr_pad, mm_i_scale, mm_i_zp, mm_w_scale, mm_w_zp, mm_o_scale,
      mm_o_zp, gemm_m, gemm_k, gemm_n, 16, 256, 8);
  binary_op_wts_write(mm_wts, wts_ptr, rtp_ptr);
  return mm_wts.size();
}

// transpose from (oc,ic,hw) to (hw, ic, oc)
inline std::vector<uint8_t> transpose_conv_weight(uint8_t* wts_ptr, int oc,
                                                  int ic, int hw) {
  std::vector<uint8_t> output(hw * ic * oc);

  for (std::size_t o = 0; o < oc; ++o) {
    for (std::size_t i = 0; i < ic; ++i) {
      for (std::size_t h = 0; h < hw; ++h) {
        output[h * (ic * oc) + i * oc + o] =
            wts_ptr[o * (ic * hw) + i * hw + h];
      }
    }
  }
  return output;
}

struct Float_Shifted_Int32 {
  int32_t best_int;
  uint16_t best_shift_val;
};
inline Float_Shifted_Int32 find_closest_shifted_int32(float float_val) {
  Float_Shifted_Int32 result;
  const int TMP_INT32_MAX = 8388607;
  float prev_rel_err = 1e9;
  float curr_float_val = float_val;
  float best_float_val = 0;
  uint16_t shift_val = 0;
  int32_t best_int;
  int32_t closest_curr_int;
  uint16_t best_shift_val;
  float cur_rel_err;
  while (curr_float_val <= TMP_INT32_MAX) {
    closest_curr_int = std::round(curr_float_val);
    cur_rel_err =
        std::abs(float_val - closest_curr_int / std::pow(2.0, shift_val)) /
        float_val;
    if (cur_rel_err < prev_rel_err) {
      prev_rel_err = cur_rel_err;
      best_float_val = static_cast<double>(closest_curr_int >> shift_val);
      best_shift_val = shift_val;
      best_int = closest_curr_int;
    }
    curr_float_val *= 2;
    shift_val += 1;
  }
  result.best_int = best_int;
  result.best_shift_val = best_shift_val;
  return result;
}

inline void lp_c0_patch(uint32_t* lp, uint32_t* c0, uint32_t* lp_out,
                        uint32_t* c0_out, uint8_t* weights,
                        int32_t weights_out_ch, int32_t weights_in_ch,
                        int32_t weights_ky, int32_t weights_kx, int32_t* bias,
                        float a_dq_xscale, int64_t a_dq_xzero_pt,
                        float w_dq_xscale, int64_t w_dq_xzero_pt,
                        float a_q_yscale, int64_t a_q_yzero_pt) {
  // get matmul_shift
  int tmp_product =
      std::ceil(std::log2(weights_in_ch * weights_ky * weights_kx));
  int tmp_value = 25 + tmp_product - 32;
  tmp_value = max(tmp_value, 0);
  int matmul_shift = min(tmp_value, 7);

  // c2 calculation
  float c2_coeff = (a_dq_xscale * w_dq_xscale) / a_q_yscale;

  Float_Shifted_Int32 tmp = find_closest_shifted_int32(c2_coeff);
  int64_t c2_coeff_prime = (int64_t)tmp.best_int;
  uint16_t shft_c2 = tmp.best_shift_val;

  // c3 calculation
  int64_t num_weights_unrolled = weights_ky * weights_kx * weights_in_ch;
  int64_t c3_coeff_offset = (int64_t)(-a_dq_xzero_pt * num_weights_unrolled);
  int64_t c3_coeff_scale = (int64_t)(-c2_coeff_prime * (int64_t)w_dq_xzero_pt);

  // c1 calculation
  std::vector<int64_t> c1_coeff(weights_out_ch);
  for (int axis_0 = 0; axis_0 < weights_out_ch; axis_0++) {
    int64_t tmp_sum = 0;
    for (int axis_1 = 0; axis_1 < weights_in_ch; axis_1++)
      for (int axis_2 = 0; axis_2 < weights_ky; axis_2++)
        for (int axis_3 = 0; axis_3 < weights_kx; axis_3++) {
          // tmp_sum +=(uint64_t)weights[axis_0][axis_1][axis_2][axis_3];
          int tmp_idx = axis_0 * (weights_in_ch * weights_ky * weights_kx) +
                        axis_1 * (weights_ky * weights_kx) +
                        axis_2 * (weights_kx) + axis_3;
          uint16_t tmp_data =
              (weights[axis_0 * (weights_in_ch * weights_ky * weights_kx) +
                       axis_1 * (weights_ky * weights_kx) +
                       axis_2 * (weights_kx) + axis_3]);
          tmp_sum += (uint64_t)tmp_data;
        }
    c1_coeff[axis_0] = ((double)(-a_dq_xzero_pt)) * c2_coeff_prime * tmp_sum +
                       c2_coeff_prime * bias[axis_0] +
                       ((int64_t)a_q_yzero_pt << shft_c2);
  }

  // C0, C1, C2
  std::vector<int64_t> C0(weights_out_ch);
  for (int i = 0; i < weights_out_ch; i++) {
    C0[i] = c3_coeff_scale * c3_coeff_offset + c1_coeff[i];
    uint32_t tmp_data_c0 = static_cast<uint32_t>(C0[i] & 0xFFFFFFFF);
    c0_out[2 * i] = tmp_data_c0;
    tmp_data_c0 = static_cast<uint32_t>((C0[i] >> 32) & 0xFFFFFFFF);
    c0_out[2 * i + 1] = tmp_data_c0;
  }
  int64_t C1 = c3_coeff_scale;
  int64_t C2 = c2_coeff_prime << matmul_shift;
  int64_t shift_conv = matmul_shift;
  int64_t shift_final = shft_c2;

  for (int i = 0; i < 16; i++) {
    lp_out[i] = lp[i];
  }
  lp_out[4] = (lp_out[4] & 0xFFFF0000) | ((shift_final & 0xFF) << 8) |
              (shift_conv & 0xFF);
  lp_out[11] = C1;
  lp_out[12] = C2;
}

inline std::vector<uint64_t>
conv_generate_wts(uint8_t* weights, int32_t* bias, float ifm_scale,
                  int64_t ifm_zero_point, float w_scale, int64_t w_zero_point,
                  float b_scale, int64_t b_zero_point, float ofm_scale,
                  int64_t ofm_zero_point, int32_t oc, int32_t ic, int32_t h,
                  int32_t w, int32_t sv_ic, uint32_t* c0, uint32_t* lp,
                  std::string model_version = "GT_v1.2") {

  std::vector<uint8_t> result(64); // for wts zero padding
  uint8_t* w_ptr = (uint8_t*)weights;
  int32_t ICP = 8;
  int32_t OCP = 8;
  int32_t ic_tile = ICP;     // IFM tile C dimension granularity
  int32_t icg = sv_ic / ICP; // IFM tile C dimension group
  // OFM related
  int32_t sv_oc = 16;        // sub-volume OFM C dimension
  int32_t oc_tile = OCP;     // OFM tile C dimension granularity
  int32_t ocg = sv_oc / OCP; // OFM tile C dimension group
  if (model_version == "GT_v1.3") {
    int32_t weights_out_ch = oc;
    int32_t weights_in_ch = ic;
    int32_t weights_ky = h;
    int32_t weights_kx = w;
    uint32_t* c0_out = new uint32_t[weights_out_ch * 2];
    uint32_t* lp_out = new uint32_t[16];
    lp_c0_patch(lp, c0, lp_out, c0_out, w_ptr, weights_out_ch, weights_in_ch,
                weights_ky, weights_kx, bias, ifm_scale, ifm_zero_point,
                w_scale, w_zero_point, ofm_scale, ofm_zero_point);
    auto wts_data = wts_gen_conv(lp_out, (int64_t*)c0_out, w_ptr, h, w, ic, oc,
                                 ic_tile, icg, oc_tile, ocg);
    delete[] c0_out;
    delete[] lp_out;
    return wts_data;
  } else {
    auto wts_data = wts_gen_conv(lp, (int64_t*)c0, w_ptr, h, w, ic, oc, ic_tile,
                                 icg, oc_tile, ocg);
    return wts_data;
  }
}

inline size_t GT_CONV_WTS_convert(
    const std::unordered_map<std::string, flexmlrt::client::ErtIoTypeNew>& wts_,
    int8_t*& wts_ptr, const std::string& conv_i_scale_name,
    const std::string& conv_i_zp_name, const std::string& conv_w_scale_name,
    const std::string& conv_w_zp_name, const std::string& conv_b_scale_name,
    const std::string& conv_b_zp_name, const std::string& conv_o_scale_name,
    const std::string& conv_o_zp_name, const std::string& conv_w_name,
    const std::string& conv_b_name, int32_t oc, int32_t ic, int32_t h,
    int32_t w, int32_t sv_ic, uint32_t* c0, uint32_t* lp,
    std::string model_version = "GT_v1.2") {
  float conv_i_scale = 1.0;
  float conv_w_scale = 1.0;
  float conv_b_scale = 1.0;
  float conv_o_scale = 1.0;
  uint16_t conv_i_zp = 0;
  uint8_t conv_w_zp = 0;
  int32_t conv_b_zp = 0;
  uint16_t conv_o_zp = 0;
  get_s_zp(conv_i_scale_name, conv_i_scale, conv_i_zp_name, conv_i_zp, wts_,
           model_version);
  get_s_zp(conv_w_scale_name, conv_w_scale, conv_w_zp_name, conv_w_zp, wts_,
           model_version);
  get_s_zp(conv_b_scale_name, conv_b_scale, conv_b_zp_name, conv_b_zp, wts_,
           model_version);
  get_s_zp(conv_o_scale_name, conv_o_scale, conv_o_zp_name, conv_o_zp, wts_,
           model_version);
  uint8_t* conv_w_ptr = (uint8_t*)(wts_.at(conv_w_name).data);
  int32_t* conv_b_ptr = (int32_t*)(wts_.at(conv_b_name).data);
  auto conv_w_shape = wts_.at(conv_w_name).shape;
  auto conv_b_shape = wts_.at(conv_b_name).shape;

  std::vector<uint64_t> conv_wts = conv_generate_wts(
      conv_w_ptr, conv_b_ptr, conv_i_scale, conv_i_zp, conv_w_scale, conv_w_zp,
      conv_b_scale, conv_b_zp, conv_o_scale, conv_o_zp, oc, ic, h, w, sv_ic, c0,
      lp, model_version);
  memcpy(wts_ptr, (int8_t*)(conv_wts.data()),
         conv_wts.size() * sizeof(uint64_t));
  wts_ptr += conv_wts.size() * sizeof(uint64_t);
  return conv_wts.size() * sizeof(uint64_t);
}

inline size_t GT_ADD_WTS_convert(
    const std::unordered_map<std::string, flexmlrt::client::ErtIoTypeNew>& wts_,
    int8_t*& wts_ptr, int8_t*& rtp_ptr, const std::string& add_i_scale_name,
    const std::string& add_i_zp_name, const std::string& add_w_scale_name,
    const std::string& add_w_zp_name, const std::string& add_o_scale_name,
    const std::string& add_o_zp_name, const std::string& add_w_name) {

  float add_w_scale = 1.0;
  float add_i_scale = 1.0;
  float add_o_scale = 1.0;
  uint16_t add_w_zp = 0;
  uint16_t add_i_zp = 0;
  uint16_t add_o_zp = 0;
  get_s_zp(add_i_scale_name, add_i_scale, add_i_zp_name, add_i_zp, wts_);
  get_s_zp(add_w_scale_name, add_w_scale, add_w_zp_name, add_w_zp, wts_);
  get_s_zp(add_o_scale_name, add_o_scale, add_o_zp_name, add_o_zp, wts_);
  if (wts_.find(add_w_name) == wts_.end()) {
    // std::cout << add_w_name << " not found " << std::endl;
  }
  uint16_t* add_w_ptr = (uint16_t*)(wts_.at(add_w_name).data);
  auto add_w_shape = wts_.at(add_w_name).shape;
  size_t add_wts_size = std::accumulate(add_w_shape.begin(), add_w_shape.end(),
                                        1, std::multiplies<size_t>());
  size_t add_ouput_size = add_wts_size * 25; // hardcode broadcast
  std::vector<uint8_t> add_wts =
      add_generate_wts(add_w_ptr, add_ouput_size, add_wts_size, add_i_scale,
                       add_i_zp, add_w_scale, add_w_zp, add_o_scale, add_o_zp);
  binary_op_wts_write(add_wts, wts_ptr, rtp_ptr);
  return add_wts.size();
}

inline size_t GT_ADD_WTS_QDQ_convert(
    const std::unordered_map<std::string, flexmlrt::client::ErtIoTypeNew>& wts_,
    int8_t*& wts_ptr, int8_t*& rtp_ptr, const std::string& add_i_scale_name,
    const std::string& add_i_zp_name, const std::string& add_w_scale_name,
    const std::string& add_w_zp_name, const std::string& add_o_scale_name,
    const std::string& add_o_zp_name, const std::string& add_w_name,
    const int32_t tensor_size, const std::string model_version = "GT_v1.2") {

  float add_w_scale = 1.0;
  float add_i_scale = 1.0;
  float add_o_scale = 1.0;
  uint16_t add_w_zp = 0;
  uint16_t add_i_zp = 0;
  uint16_t add_o_zp = 0;
  get_s_zp(add_i_scale_name, add_i_scale, add_i_zp_name, add_i_zp, wts_,
           model_version);
  get_s_zp(add_w_scale_name, add_w_scale, add_w_zp_name, add_w_zp, wts_,
           model_version);
  get_s_zp(add_o_scale_name, add_o_scale, add_o_zp_name, add_o_zp, wts_,
           model_version);
  // printf("add WTS QDQ param %f %d %f %d %f %d\n", add_i_scale, add_i_zp,
  // add_w_scale, add_w_zp, add_o_scale, add_o_zp);

  AddQDQParams params = add_calc_qdq_params(add_i_scale, add_i_zp, add_w_scale,
                                            add_w_zp, add_o_scale, add_o_zp);

  std::vector<uint8_t> RTP = add_get_RTP(tensor_size, params);
  binary_op_wts_write(RTP, wts_ptr, rtp_ptr);
  return RTP.size();
}
struct MulQDQParams {
  int coeff0;
  int c0_sft;
  int coeff1;
  int c1_sft;
  int ifm1_zp;
  int ifm2_zp;
};

inline MulQDQParams mul_calc_qdq_params(double ifm1_scale, int ifm1_zero_point,
                                        double ifm2_scale, int ifm2_zero_point,
                                        double ofm_scale, int ofm_zero_point) {
  MulQDQParams params;
  auto fp_bits_to_i32 = [](float f) {
    return (*reinterpret_cast<int*>(&f)) & 0x7fffffff;
  }; // remove sign bit
  auto get_sft_from_i32_rep = [](int i) {
    return 127 - (((i >> 23) & 255) + 1) + (8 * sizeof(int) - 2);
  };

  double C0 = ifm1_scale * ifm2_scale / ofm_scale;
  int C0_int_rep = fp_bits_to_i32(C0);
  params.c0_sft = get_sft_from_i32_rep(C0_int_rep);
  params.coeff0 = int(C0 * (1LL << params.c0_sft));

  double C1 = C0 * ifm1_zero_point * ifm2_zero_point + ofm_zero_point;
  int C1_int_rep = fp_bits_to_i32(C1);
  params.c1_sft = get_sft_from_i32_rep(C1_int_rep);
  params.coeff1 = int(C1 * (1LL << params.c1_sft));

  params.ifm1_zp = ifm1_zero_point;
  params.ifm2_zp = ifm2_zero_point;
  return params;
}

inline std::vector<uint8_t> mul_get_RTP(int tensor_size, int eff_dim,
                                        int pad_dim,
                                        const MulQDQParams& params) {
  std::vector<uint8_t> RTP(64, 0);
  RTP[0] = tensor_size & 0xFF;
  RTP[1] = (tensor_size >> 8) & 0xFF;
  RTP[2] = (tensor_size >> 16) & 0xFF;
  RTP[3] = (tensor_size >> 24) & 0xFF;
  RTP[4] = params.coeff0 & 0xFF;
  RTP[5] = (params.coeff0 >> 8) & 0xFF;
  RTP[6] = (params.coeff0 >> 16) & 0xFF;
  RTP[7] = (params.coeff0 >> 24) & 0xFF;
  RTP[8] = params.coeff1 & 0xFF;
  RTP[9] = (params.coeff1 >> 8) & 0xFF;
  RTP[10] = (params.coeff1 >> 16) & 0xFF;
  RTP[11] = (params.coeff1 >> 24) & 0xFF;
  RTP[12] = params.ifm1_zp & 0xFF;
  RTP[13] = (params.ifm1_zp >> 8) & 0xFF;
  RTP[14] = params.ifm2_zp & 0xFF;
  RTP[15] = (params.ifm2_zp >> 8) & 0xFF;
  RTP[16] = params.c0_sft;
  RTP[17] = params.c1_sft;
  RTP[18] = 0;
  RTP[19] = 0;
  RTP[20] = 0;
  RTP[21] = 0;
  RTP[22] = 0;
  RTP[23] = 0;
  RTP[24] = (tensor_size / (4096 * 8)) & 0xFF;
  RTP[25] = ((tensor_size / (4096 * 8)) >> 8) & 0xFF;
  RTP[26] = ((tensor_size / (4096 * 8)) >> 16) & 0xFF;
  RTP[27] = ((tensor_size / (4096 * 8)) >> 24) & 0xFF;
  RTP[28] = (eff_dim)&0xFF;
  RTP[29] = (eff_dim >> 8) & 0xFF;
  RTP[30] = (eff_dim >> 16) & 0xFF;
  RTP[31] = (eff_dim >> 24) & 0xFF;
  RTP[32] = pad_dim & 0xFF;
  RTP[33] = (pad_dim >> 8) & 0xFF;
  RTP[34] = (pad_dim >> 16) & 0xFF;
  RTP[35] = (pad_dim >> 24) & 0xFF;
  return RTP;
}

inline std::vector<uint8_t>
mul_generate_wts(uint16_t* wts_ptr, const std::vector<size_t>& wts_shape,
                 float ifm1_scale, int ifm1_zero_point, float ifm2_scale,
                 int ifm2_zero_point, float ofm_scale, int ofm_zero_point) {
  size_t wts_size = std::accumulate(wts_shape.begin(), wts_shape.end(), 1,
                                    std::multiplies<size_t>());

  MulQDQParams params =
      mul_calc_qdq_params(ifm1_scale, ifm1_zero_point, ifm2_scale,
                          ifm2_zero_point, ofm_scale, ofm_zero_point);

  int tensor_size = 1;
  int eff_dim = 1;
  int pad_dim = 1;
  if (wts_size == 1) { // scalar mul
    tensor_size = 8 * 25 * 64;
    eff_dim = 64;
    pad_dim = 64;
  } else if (wts_size == 8 * 25 * 475) { // elem mul with padding
    tensor_size = 8 * 25 * 480;
    eff_dim = 475;
    pad_dim = 480;
  }
  std::vector<uint8_t> RTP = mul_get_RTP(tensor_size, eff_dim, pad_dim, params);
  std::vector<uint8_t> result(64 + tensor_size * 2,
                              ifm2_zero_point); // for wts zero padding
  uint8_t* result_ptr = result.data();

  memcpy(result_ptr, RTP.data(), 64);
  result_ptr += 64;

  if (wts_size == 1) {
    for (int i = 0; i < tensor_size; i++) {
      ((uint16_t*)result_ptr)[i] = wts_ptr[0];
    }
  } else if (wts_size == 8 * 25 * 475) {
    for (int i = 0; i < 8 * 25; i++) {
      memcpy(result_ptr, wts_ptr, 475 * 2);
      result_ptr += 480 * sizeof(uint16_t);
      wts_ptr += 475 * sizeof(uint16_t);
    }
  }
  return result;
}
inline size_t GT_MUL_WTS_convert(
    const std::unordered_map<std::string, flexmlrt::client::ErtIoTypeNew>& wts_,
    int8_t*& wts_ptr, int8_t*& rtp_ptr, const std::string& mul_i_scale_name,
    const std::string& mul_i_zp_name, const std::string& mul_w_scale_name,
    const std::string& mul_w_zp_name, const std::string& mul_o_scale_name,
    const std::string& mul_o_zp_name, const std::string& mul_w_name,
    const std::string model_version = "GT_v1.2") {

  float mul_w_scale = 1.0;
  float mul_i_scale = 1.0;
  float mul_o_scale = 1.0;
  uint16_t mul_w_zp = 0;
  uint16_t mul_i_zp = 0;
  uint16_t mul_o_zp = 0;
  get_s_zp(mul_i_scale_name, mul_i_scale, mul_i_zp_name, mul_i_zp, wts_,
           model_version);
  get_s_zp(mul_w_scale_name, mul_w_scale, mul_w_zp_name, mul_w_zp, wts_,
           model_version);
  get_s_zp(mul_o_scale_name, mul_o_scale, mul_o_zp_name, mul_o_zp, wts_,
           model_version);

  uint16_t* mul_w_ptr = (uint16_t*)(wts_.at(mul_w_name).data);
  auto mul_w_shape = wts_.at(mul_w_name).shape;

  std::vector<uint8_t> mul_wts =
      mul_generate_wts(mul_w_ptr, mul_w_shape, mul_i_scale, mul_i_zp,
                       mul_w_scale, mul_w_zp, mul_o_scale, mul_o_zp);
  binary_op_wts_write(mul_wts, wts_ptr, rtp_ptr);
  return mul_wts.size();
}

inline std::vector<uint8_t> front_sub_get_RTP(int tensor_size, int w,
                                              const AddQDQParams& params) {
  std::vector<uint8_t> RTP(64, 0);
  RTP[0] = 1;
  RTP[1] = 1;
  RTP[2] = 1;
  RTP[3] = 1;
  RTP[4] = 560 & 0xFF;
  RTP[5] = (560 >> 8) & 0xFF;
  RTP[6] = (560 >> 16) & 0xFF;
  RTP[7] = (560 >> 24) & 0xFF;
  RTP[8] = w & 0xFF;
  RTP[9] = (w >> 8) & 0xFF;
  RTP[10] = (w >> 16) & 0xFF;
  RTP[11] = (w >> 24) & 0xFF;
  RTP[12] = params.ifm1_coeff & 0xFF;
  RTP[13] = (params.ifm1_coeff >> 8) & 0xFF;
  RTP[14] = (params.ifm1_coeff >> 16) & 0xFF;
  RTP[15] = (params.ifm1_coeff >> 24) & 0xFF;
  RTP[16] = params.ifm2_coeff & 0xFF;
  RTP[17] = (params.ifm2_coeff >> 8) & 0xFF;
  RTP[18] = (params.ifm2_coeff >> 16) & 0xFF;
  RTP[19] = (params.ifm2_coeff >> 24) & 0xFF;
  RTP[20] = params.zero_point_coeff & 0xFF;
  RTP[21] = (params.zero_point_coeff >> 8) & 0xFF;
  RTP[22] = (params.zero_point_coeff >> 16) & 0xFF;
  RTP[23] = (params.zero_point_coeff >> 24) & 0xFF;
  RTP[24] = params.ofm_shift;
  RTP[25] = params.ifm1_shift;
  RTP[26] = params.ifm2_shift;
  RTP[27] = params.zero_point_shift;
  RTP[28] = (560 / 16) & 0xFF;
  RTP[29] = (w / 16) & 0xFF;
  RTP[30] = 0;
  RTP[31] = 0;
  RTP[32] = 0;
  RTP[33] = 0;
  RTP[34] = 0;
  RTP[35] = 0;
  RTP[36] = (9) & 0xFF;
  return RTP;
}

inline std::vector<uint8_t> front_mul_get_RTP(int tensor_size,
                                              const MulQDQParams& params) {
  std::vector<uint8_t> RTP(64, 0);
  RTP[0] = (560) & 0xFF;
  RTP[1] = (560 >> 8) & 0xFF;
  RTP[2] = (560 >> 16) & 0xFF;
  RTP[3] = (560 >> 24) & 0xFF;
  RTP[4] = params.coeff0 & 0xFF;
  RTP[5] = (params.coeff0 >> 8) & 0xFF;
  RTP[6] = (params.coeff0 >> 16) & 0xFF;
  RTP[7] = (params.coeff0 >> 24) & 0xFF;
  RTP[8] = params.coeff1 & 0xFF;
  RTP[9] = (params.coeff1 >> 8) & 0xFF;
  RTP[10] = (params.coeff1 >> 16) & 0xFF;
  RTP[11] = (params.coeff1 >> 24) & 0xFF;
  RTP[12] = params.ifm1_zp & 0xFF;
  RTP[13] = (params.ifm1_zp >> 8) & 0xFF;
  RTP[14] = params.ifm2_zp & 0xFF;
  RTP[15] = (params.ifm2_zp >> 8) & 0xFF;
  RTP[16] = params.c0_sft;
  RTP[17] = params.c1_sft;
  RTP[18] = 0;
  RTP[19] = 0;
  RTP[20] = (1) & 0xFF;
  RTP[21] = (1 >> 8) & 0xFF;
  RTP[22] = 0;
  RTP[23] = 0;
  RTP[24] = (1) & 0xFF;
  RTP[25] = (1 >> 8) & 0xFF;
  RTP[26] = (80 / 16) & 0xFF;
  RTP[27] = (80 / 16 >> 8) & 0xFF;
  RTP[28] = 0;
  RTP[29] = 0;
  RTP[30] = 0;
  RTP[31] = 0;
  RTP[32] = 0;
  RTP[33] = 0;
  RTP[34] = 0;
  RTP[35] = 0;
  RTP[36] = (7) & 0xFF;
  return RTP;
}

inline std::vector<uint8_t> sub_mul_generate_wts(
    uint16_t* sub_wts_ptr, const std::vector<size_t>& sub_wts_shape,
    float sub_ifm1_scale, int sub_ifm1_zero_point, float sub_ifm2_scale,
    int sub_ifm2_zero_point, float sub_ofm_scale, int sub_ofm_zero_point,
    uint16_t* mul_wts_ptr, const std::vector<size_t>& mul_wts_shape,
    float mul_ifm1_scale, int mul_ifm1_zero_point, float mul_ifm2_scale,
    int mul_ifm2_zero_point, float mul_ofm_scale, int mul_ofm_zero_point) {
  size_t wts_size = 103 * 80;
  size_t width = 80;
  std::vector<uint8_t> result((64 + width * sizeof(uint16_t)) * 2,
                              0); // for wts zero padding
  uint8_t* result_ptr = result.data();
  AddQDQParams sub_params = add_calc_qdq_params(
      sub_ifm1_scale, sub_ifm1_zero_point, sub_ifm2_scale, sub_ifm2_zero_point,
      sub_ofm_scale, sub_ofm_zero_point);
  std::vector<uint8_t> sub_RTP = front_sub_get_RTP(wts_size, width, sub_params);
  memcpy(result_ptr, sub_RTP.data(), 64);
  result_ptr += 64;
  memcpy(result_ptr, sub_wts_ptr, width * sizeof(uint16_t));
  result_ptr += width * sizeof(uint16_t);

  MulQDQParams params = mul_calc_qdq_params(mul_ifm1_scale, mul_ifm1_zero_point,
                                            mul_ifm2_scale, mul_ifm2_zero_point,
                                            mul_ofm_scale, mul_ofm_zero_point);
  std::vector<uint8_t> RTP = front_mul_get_RTP(wts_size, params);

  memcpy(result_ptr, RTP.data(), 64);
  result_ptr += 64;
  memcpy(result_ptr, mul_wts_ptr, width * sizeof(uint16_t));
  return result;
}

inline size_t GT_SUB_MUL_WTS_convert(
    const std::unordered_map<std::string, flexmlrt::client::ErtIoTypeNew>& wts_,
    int8_t*& wts_ptr, const std::string& sub_i_scale_name,
    const std::string& sub_i_zp_name, const std::string& sub_w_scale_name,
    const std::string& sub_w_zp_name, const std::string& sub_o_scale_name,
    const std::string& sub_o_zp_name, const std::string& sub_w_name,
    const std::string& mul_i_scale_name, const std::string& mul_i_zp_name,
    const std::string& mul_w_scale_name, const std::string& mul_w_zp_name,
    const std::string& mul_o_scale_name, const std::string& mul_o_zp_name,
    const std::string& mul_w_name,
    const std::string model_version = "GT_v1.2") {

  float sub_w_scale = 1.0;
  float sub_i_scale = 1.0;
  float sub_o_scale = 1.0;
  uint16_t sub_w_zp = 0;
  uint16_t sub_i_zp = 0;
  uint16_t sub_o_zp = 0;
  get_s_zp(sub_i_scale_name, sub_i_scale, sub_i_zp_name, sub_i_zp, wts_,
           model_version);
  get_s_zp(sub_w_scale_name, sub_w_scale, sub_w_zp_name, sub_w_zp, wts_,
           model_version);
  get_s_zp(sub_o_scale_name, sub_o_scale, sub_o_zp_name, sub_o_zp, wts_,
           model_version);

  uint16_t* sub_w_ptr = (uint16_t*)(wts_.at(sub_w_name).data);
  auto sub_w_shape = wts_.at(sub_w_name).shape;

  float mul_w_scale = 1.0;
  float mul_i_scale = 1.0;
  float mul_o_scale = 1.0;
  uint16_t mul_w_zp = 0;
  uint16_t mul_i_zp = 0;
  uint16_t mul_o_zp = 0;
  get_s_zp(mul_i_scale_name, mul_i_scale, mul_i_zp_name, mul_i_zp, wts_,
           model_version);
  get_s_zp(mul_w_scale_name, mul_w_scale, mul_w_zp_name, mul_w_zp, wts_,
           model_version);
  get_s_zp(mul_o_scale_name, mul_o_scale, mul_o_zp_name, mul_o_zp, wts_,
           model_version);

  uint16_t* mul_w_ptr = (uint16_t*)(wts_.at(mul_w_name).data);
  auto mul_w_shape = wts_.at(mul_w_name).shape;

  std::vector<uint8_t> mul_wts = sub_mul_generate_wts(
      sub_w_ptr, sub_w_shape, sub_i_scale, sub_i_zp, sub_w_scale, sub_w_zp,
      sub_o_scale, sub_o_zp, mul_w_ptr, mul_w_shape, mul_i_scale, mul_i_zp,
      mul_w_scale, mul_w_zp, mul_o_scale, mul_o_zp);
  memcpy(wts_ptr, mul_wts.data(), mul_wts.size());
  wts_ptr += mul_wts.size();
  return mul_wts.size();
}

inline size_t GT_MUL_WTS_QDQ_convert(
    const std::unordered_map<std::string, flexmlrt::client::ErtIoTypeNew>& wts_,
    int8_t*& wts_ptr, int8_t*& rtp_ptr, const std::string& mul_i_scale_name,
    const std::string& mul_i_zp_name, const std::string& mul_w_scale_name,
    const std::string& mul_w_zp_name, const std::string& mul_o_scale_name,
    const std::string& mul_o_zp_name, const std::string& mul_w_name,
    const int32_t tensor_size, const std::string model_version = "GT_v1.2") {

  float mul_w_scale = 1.0;
  float mul_i_scale = 1.0;
  float mul_o_scale = 1.0;
  uint16_t mul_w_zp = 0;
  uint16_t mul_i_zp = 0;
  uint16_t mul_o_zp = 0;
  get_s_zp(mul_i_scale_name, mul_i_scale, mul_i_zp_name, mul_i_zp, wts_,
           model_version);
  get_s_zp(mul_w_scale_name, mul_w_scale, mul_w_zp_name, mul_w_zp, wts_,
           model_version);
  get_s_zp(mul_o_scale_name, mul_o_scale, mul_o_zp_name, mul_o_zp, wts_,
           model_version);

  // printf("mul qdq param %f %d %f %d %f %d\n", mul_i_scale, mul_i_zp,
  // mul_w_scale,
  //   mul_w_zp, mul_o_scale, mul_o_zp);
  MulQDQParams params = mul_calc_qdq_params(mul_i_scale, mul_i_zp, mul_w_scale,
                                            mul_w_zp, mul_o_scale, mul_o_zp);
  std::vector<uint8_t> RTP = mul_get_RTP(tensor_size, 475, 480, params);
  binary_op_wts_write(RTP, wts_ptr, rtp_ptr);
  return RTP.size();
}

inline std::vector<uint8_t> batch_matmul_A16A16_generate_wts(
    float ifm1_scale, int64_t ifm1_zero_point, float ifm2_scale,
    int64_t ifm2_zero_point, float ofm_scale, int64_t ofm_zero_point,
    uint32_t gemm_m, uint32_t gemm_k, uint32_t gemm_n, bool transpose_b,
    uint32_t sv_M, uint32_t sv_K, uint32_t sv_N) {
  const uint32_t NUM_COLS = 4;
  const uint32_t NUM_ROWS = 4;
  uint32_t matmul_shift = 0;
  matmul_shift =
      gt_min(gt_max((33 + std::ceil(std::log2(gemm_k)) - 32), 0), 15);

  float c2_fp;
  c2_fp = (float)((ifm1_scale * ifm2_scale) / ofm_scale);
  auto res0 = find_closest_shifted_int16(c2_fp);
  int64_t c2 = res0.first;
  int64_t shft_c2 = res0.second;
  // std::cout << "[qdq_matmul_uint16_uint16]: C2 " << c2 << " C2_sft " <<
  // shft_c2 << std::endl;
  int64_t c3_coeff_scale = -c2 * ifm2_zero_point;
  int32_t C3 = (c2 << matmul_shift);
  int32_t C2 = c3_coeff_scale;
  int32_t c1 = ((-ifm1_zero_point) * c2);

  int64_t c0 = ((ofm_zero_point << shft_c2) +
                int64_t(int64_t(ifm1_zero_point) * int64_t(ifm2_zero_point) *
                        int64_t(gemm_k) * c2));
  uint32_t iters =
      gemm_n == 512
          ? ((transpose_b << 24) | (((gemm_m + sv_M - 1) / sv_M) << 16) |
             ((gemm_k / sv_K) << 8) | (gemm_n / sv_N / NUM_COLS / NUM_ROWS))
          : ((transpose_b << 24) | (((gemm_m + sv_M - 1) / sv_M) << 16) |
             ((gemm_k / sv_K) << 8) | (gemm_n / sv_N / NUM_ROWS));
  std::vector<uint32_t> RTP = {sv_M,
                               sv_K,
                               sv_N,
                               iters,
                               0x2000,
                               0x4800,
                               0x3800,
                               uint32_t(matmul_shift | (shft_c2 << 16)),
                               0x3c00,
                               0x4400,
                               ((uint32_t*)&c0)[0],
                               ((uint32_t*)&c0)[1],
                               uint32_t(c1),
                               uint32_t(C2),
                               uint32_t(C3),
                               9};
  std::vector<uint8_t> res_uint8(gemm_k == 512 ? (64 + 12800 + 37888) : 64, 0);
  uint8_t* res_ptr = res_uint8.data();
  memcpy(res_ptr, RTP.data(), 64);
  res_ptr += 64;
  if (gemm_k == 512) { // need padding (MatMul_2)
    // 8x25x480 -> 8x25x512
    std::vector<std::uint16_t> padding_A(6400, ifm1_zero_point);
    memcpy(res_ptr, padding_A.data(), 12800);
    res_ptr += 12800;
    // 8x475x64 -> 8x512x64
    std::vector<std::uint16_t> padding_B(18944, ifm2_zero_point);
    memcpy(res_ptr, padding_B.data(), 37888);
    res_ptr += 37888;
  }
  return res_uint8;
}

// default uint16_t wts dtype to for gt1.2
template <typename wts_dtype = uint16_t>
inline size_t GT_BMM_WTS_convert(
    const std::unordered_map<std::string, flexmlrt::client::ErtIoTypeNew>& wts_,
    int8_t*& wts_ptr, int8_t*& rtp_ptr, const std::string& mm_i_scale_name,
    const std::string& mm_i_zp_name, const std::string& mm_w_scale_name,
    const std::string& mm_w_zp_name, const std::string& mm_o_scale_name,
    const std::string& mm_o_zp_name, const std::string& mm_w_name,
    bool transpose_b, uint32_t gemm_M, uint32_t gemm_K, uint32_t gemm_N,
    uint32_t sv_M, uint32_t sv_K, uint32_t sv_N,
    const std::string model_version = "GT_v1.2") {
  // MatMul
  float mm_i_scale = 1.0;
  float mm_w_scale = 1.0;
  float mm_o_scale = 1.0;
  uint16_t mm_i_zp = 0;
  wts_dtype mm_w_zp = 0;
  uint16_t mm_o_zp = 0;
  get_s_zp(mm_i_scale_name, mm_i_scale, mm_i_zp_name, mm_i_zp, wts_,
           model_version);
  get_s_zp(mm_w_scale_name, mm_w_scale, mm_w_zp_name, mm_w_zp, wts_,
           model_version);
  get_s_zp(mm_o_scale_name, mm_o_scale, mm_o_zp_name, mm_o_zp, wts_,
           model_version);

  std::vector<uint8_t> mm_wts = batch_matmul_A16A16_generate_wts(
      mm_i_scale, mm_i_zp, mm_w_scale, mm_w_zp, mm_o_scale, mm_o_zp, gemm_M,
      gemm_K, gemm_N, transpose_b, sv_M, sv_K, sv_N);
  uint32_t* tmp_rtp = (uint32_t*)rtp_ptr;
  // for (int i = 0; i < 16; ++)
  //   printf("bmm_1 rtp [%d] = %d\n", i, tmp_rtp[i]);
  binary_op_wts_write(mm_wts, wts_ptr, rtp_ptr);
  return mm_wts.size();
}

inline std::vector<uint8_t>
softmax_generate_wts(float ifm_scale, int64_t ifm_zero_point, float ofm_scale,
                     int64_t ofm_zero_point, uint32_t K, uint32_t K_valid) {
  const size_t BF16_BYTES = 2;
  union U {
    float f;
    uint16_t ui16[2];
    uint32_t ui32;
    U(float fp) : f(fp){};
  };
  auto float_to_bfloat16 = [](float x) {
    uint32_t i;
    uint8_t* src = (uint8_t*)&x;
    uint8_t* tmp = (uint8_t*)&i;
    // copy float to uint32_t
    tmp[0] = src[0];
    tmp[1] = src[1];
    tmp[2] = src[2];
    tmp[3] = src[3];
    // round to nearest even
    uint32_t lsb = (i >> 16) & 0x1;
    uint32_t bias = 0x7fff + lsb;
    i += bias;
    // extract upper half of input
    uint16_t y = uint16_t(i >> 16);
    return y;
  };
  uint32_t loop_cnt = int(K / 16);
  uint32_t num_remain = K_valid - 16 * (loop_cnt - 1);
  uint32_t tile_cnt = 1;
  uint32_t lut_size = 4096;
  std::vector<uint32_t> RTP = {0,
                               0,
                               num_remain,
                               tile_cnt,
                               loop_cnt,
                               lut_size,
                               U(1.0 / ofm_scale).ui32,
                               uint32_t(ofm_zero_point),
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0};

  std::vector<float> lut;
  int bandwidth = 256;
  int align = 8;
  int lut_bytes = 2;
  int stride = bandwidth / align / lut_bytes;
  int i_stride = 512 / (2 * stride);

  lut.resize(4 * 512);
  for (int n = 0; n < 2; n++) {
    for (int i = 0; i < i_stride; i++) {
      for (int m = 0; m < 2; m++) {
        for (int k = 0; k < stride; k++) {
          unsigned char x = static_cast<unsigned char>(k + i * stride);
          float lut_lsb = std::exp(x * ifm_scale);
          float lut_msb = std::exp(static_cast<char>(x) * ifm_scale * 256.0);
          lut[n * 1024 + i * 2 * stride + m * stride + k] = lut_lsb;
          lut[n * 1024 + i * 2 * stride + m * stride + k + 512] = lut_msb;
        }
      }
    }
  }

  lut.insert(lut.end(), lut.begin(), lut.end()); // Extend the lut vector
  std::vector<uint16_t> bfloat16_lut(4 * 512 * 2);
  for (int i = 0; i < 4 * 512 * 2; i++) {
    bfloat16_lut[i] = float_to_bfloat16(lut[i]);
  }
  std::vector<uint8_t> res_uint8(64 + 4 * 512 * 2 * BF16_BYTES, 0);
  uint8_t* res_ptr = res_uint8.data();
  memcpy(res_ptr, RTP.data(), 64);
  res_ptr += 64;
  memcpy(res_ptr, bfloat16_lut.data(), 4 * 512 * 2 * BF16_BYTES);
  return res_uint8;
}

inline size_t GT_SOFTMAX_WTS_convert(
    const std::unordered_map<std::string, flexmlrt::client::ErtIoTypeNew>& wts_,
    int8_t*& wts_ptr, const std::string& sm_i_scale_name,
    const std::string& sm_i_zp_name, const std::string& sm_o_scale_name,
    const std::string& sm_o_zp_name, uint32_t K, uint32_t K_valid,
    const std::string model_version = "GT_v1.2") {

  float sm_i_scale = 1.0;
  float sm_o_scale = 1.0;
  uint16_t sm_i_zp = 0;
  uint16_t sm_o_zp = 0;
  get_s_zp(sm_i_scale_name, sm_i_scale, sm_i_zp_name, sm_i_zp, wts_,
           model_version);
  get_s_zp(sm_o_scale_name, sm_o_scale, sm_o_zp_name, sm_o_zp, wts_,
           model_version);

  std::vector<uint8_t> softmax_wts = softmax_generate_wts(
      sm_i_scale, sm_i_zp, sm_o_scale, sm_o_zp, K, K_valid);

  memcpy(wts_ptr, softmax_wts.data(), softmax_wts.size());
  wts_ptr += softmax_wts.size();
  return softmax_wts.size();
}

inline std::vector<uint8_t>
ln_generate_wts(uint8_t* s, int32_t* b, size_t K, float ifm1_scale,
                int64_t ifm1_zero_point, float s_scale, int64_t s_zero_point,
                float b_scale, int64_t b_zero_point, float ofm_scale,
                int64_t ofm_zero_point) {
  union U {
    float f;
    uint16_t ui16[2];
    uint32_t ui32;
    U(float fp) : f(fp){};
  };
  const size_t BF16_BYTES = 2;
  std::vector<uint8_t> res_uint8(64 + K * BF16_BYTES * 2, 0);

  auto float_to_bfloat16 = [](float x) {
    uint32_t i;
    uint8_t* src = (uint8_t*)&x;
    uint8_t* tmp = (uint8_t*)&i;
    // copy float to uint32_t
    tmp[0] = src[0];
    tmp[1] = src[1];
    tmp[2] = src[2];
    tmp[3] = src[3];
    // round to nearest even
    uint32_t lsb = (i >> 16) & 0x1;
    uint32_t bias = 0x7fff + lsb;
    i += bias;
    // extract upper half of input
    uint16_t y = uint16_t(i >> 16);
    return y;
  };
  auto dequant = [&float_to_bfloat16](auto* qi, int64_t zp, float scale,
                                      size_t K) {
    std::vector<uint16_t> dq_vec(K);
    for (int i = 0; i < K; i++) {
      float dq = double(qi[i] - zp) * scale;
      // dq_vec[i] = (U(dq).ui16[1]);
      dq_vec[i] = float_to_bfloat16(dq);
    }
    return dq_vec;
  };
  std::vector<uint32_t> RTP = {uint32_t(K),
                               uint32_t(K),
                               uint32_t(ifm1_zero_point),
                               U(ifm1_scale).ui16[1],
                               uint32_t(ofm_zero_point),
                               U(1.0 / ofm_scale).ui16[1],
                               U(1.0 / K).ui32,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0};
  uint8_t* res_ptr = res_uint8.data();
  memcpy(res_ptr, RTP.data(), 64);
  res_ptr += 64;
  std::vector<uint16_t> s_dq_vec = dequant(s, s_zero_point, s_scale, K);
  std::vector<uint16_t> b_dq_vec = dequant(b, b_zero_point, b_scale, K);
  memcpy(res_ptr, s_dq_vec.data(), s_dq_vec.size() * BF16_BYTES);
  res_ptr += s_dq_vec.size() * BF16_BYTES;
  memcpy(res_ptr, b_dq_vec.data(), b_dq_vec.size() * BF16_BYTES);
  return res_uint8;
}

inline size_t GT_LN_WTS_convert(
    const std::unordered_map<std::string, flexmlrt::client::ErtIoTypeNew>& wts_,
    int8_t*& wts_ptr, const std::string& ln_i_s_name,
    const std::string& ln_i_zp_name, const std::string& ln_scale_s_name,
    const std::string& ln_scale_zp_name, const std::string& ln_bias_s_name,
    const std::string& ln_bias_zp_name, const std::string& ln_o_s_name,
    const std::string& ln_o_zp_name, const std::string& ln_scale_name,
    const std::string& ln_bias_name,
    const std::string model_version = "GT_v1.2") {
  // MatMul
  float ln_i_s = 1.0;
  float ln_scale_s = 1.0;
  float ln_bias_s = 1.0;
  float ln_o_s = 1.0;

  uint16_t ln_i_zp = 0;
  uint16_t ln_o_zp = 0;
  uint8_t ln_scale_zp = 0;
  int32_t ln_bias_zp = 0;

  get_s_zp(ln_i_s_name, ln_i_s, ln_i_zp_name, ln_i_zp, wts_, model_version);
  get_s_zp(ln_scale_s_name, ln_scale_s, ln_scale_zp_name, ln_scale_zp, wts_,
           model_version);
  get_s_zp(ln_bias_s_name, ln_bias_s, ln_bias_zp_name, ln_bias_zp, wts_,
           model_version);
  get_s_zp(ln_o_s_name, ln_o_s, ln_o_zp_name, ln_o_zp, wts_, model_version);
  uint8_t* ln_scale_ptr = (uint8_t*)(wts_.at(ln_scale_name).data);
  int32_t* ln_bias_ptr = (int32_t*)(wts_.at(ln_bias_name).data);
  size_t K = 512;
  std::vector<uint8_t> ln_wts =
      ln_generate_wts(ln_scale_ptr, ln_bias_ptr, K, ln_i_s, ln_i_zp, ln_scale_s,
                      ln_scale_zp, ln_bias_s, ln_scale_zp, ln_o_s, ln_o_zp);
  memcpy(wts_ptr, ln_wts.data(), ln_wts.size());
  wts_ptr += ln_wts.size();
  return ln_wts.size();
}

inline size_t GT_QDQ_convert(
    const std::unordered_map<std::string, flexmlrt::client::ErtIoTypeNew>& wts_,
    int8_t*& wts_ptr, int8_t*& rtp_ptr, const std::string& i_s_name,
    const std::string& i_zp_name, const std::string& o_s_name,
    const std::string& o_zp_name, const std::string model_version = "GT_v1.2") {
  // MatMul
  float i_s = 1.0;
  float o_s = 1.0;

  uint16_t i_zp = 0;
  uint16_t o_zp = 0;

  get_s_zp(i_s_name, i_s, i_zp_name, i_zp, wts_, model_version);
  get_s_zp(o_s_name, o_s, o_zp_name, o_zp, wts_, model_version);

  std::vector<uint8_t> qdq(128, 0);
  i_s = 1.0 / i_s;
  uint8_t* qdq_ptr = qdq.data();
  memcpy(qdq_ptr, &i_zp, 2);
  qdq_ptr += 4;
  memcpy(qdq_ptr, (uint16_t*)(&i_s) + 1, 2);

  qdq_ptr = qdq.data() + 64;
  memcpy(qdq_ptr, &o_zp, 2);
  qdq_ptr += 4;
  memcpy(qdq_ptr, (uint16_t*)(&o_s) + 1, 2);

  memcpy(rtp_ptr, qdq.data(), qdq.size());
  rtp_ptr += qdq.size();
  return qdq.size();
}
} // namespace vaip_vaiml_custom_op
#undef gt_min
#undef gt_max

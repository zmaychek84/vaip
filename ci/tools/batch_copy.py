##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc.
##
##  Licensed under the Apache License, Version 2.0 (the "License");
##  you may not use this file except in compliance with the License.
##  You may obtain a copy of the License at
##
##  http://www.apache.org/licenses/LICENSE-2.0
##
##  Unless required by applicable law or agreed to in writing, software
##  distributed under the License is distributed on an "AS IS" BASIS,
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##  See the License for the specific language governing permissions and
##  limitations under the License.
##
import os
import sys
import shutil
import json
import subprocess
import platform
import hashlib


def copy_xco_models():
    model_list = "resnet10t mobilenetv2_050.lamb_in1k resnet14t ssl_resnet18 gluon_resnet18_v1b resnet18 mobilenetv2_100.ra_in1k efficientnet_lite0.ra_in1k lcnet_050.ra2_in1k resnet18d mnasnet_100.rmsp_in1k lcnet_075.ra2_in1k efficientnet_es.ra_in1k lcnet_100.ra2_in1k efficientnet_es_pruned.in1k ese_vovnet19b_dw swsl_resnet18 mobilenetv2_140.ra_in1k gernet_s.idstcv_in1k spnasnet_100.rmsp_in1k resnet26 resnet26d mobilenetv2_110d.ra_in1k resnet26t fbnetc_100.rmsp_in1k efficientnet_em.ra2_in1k hardcorenas_a mobilenetv3_small_050.lamb_in1k mobilenetv3_small_075.lamb_in1k mobilenetv3_small_100.lamb_in1k resnet34 gluon_resnet34_v1b tv_resnet34 resnet34d mobilenetv2_120d.ra_in1k efficientnet_el_pruned.in1k gernet_m.idstcv_in1k efficientnet_el.ra_in1k hardcorenas_b mobilenetv3_large_100.miil_in21k_ft_in1k resnet50 mobilenetv3_large_100.ra_in1k ssl_resnet50 swsl_resnet50 gluon_resnet50_v1b gluon_resnet50_v1c gluon_resnet50_v1s mobilenetv3_rw.rmsp_in1k gluon_resnet50_v1d tv_resnet50 resnet50d resnetaa50 hardcorenas_c gernet_l.idstcv_in1k xception res2net50_48w_2s repvgg_b0.rvgg_in1k gluon_inception_v3 inception_v3 tf_inception_v3 adv_inception_v3 xception41p hardcorenas_e hardcorenas_f hardcorenas_d resnet101 gluon_resnet101_v1c resnet101d gluon_resnet101_v1b tv_resnet101 gluon_resnet101_v1s gluon_resnet101_v1d ghostnet_100 gluon_xception65 vgg13 vgg11_bn vgg13_bn vgg11 xception65p vgg16 inception_v4 vgg16_bn vgg19 vgg19_bn resnet152 resnet152d tv_resnet152 gluon_resnet152_v1c gluon_resnet152_v1b gluon_resnet152_v1s gluon_resnet152_v1d resnet200d tv_densenet121 densenet121 densenet169 fbnetv3_b.ra2_in1k"
    model_list = model_list = model_list.split(" ")

    org_dir = "/proj/xsjhdstaff5/nithink/ipu_reg_setup/tests/phoenix/hw"
    det_dir = "/proj/xbb/ipu/onnx_ep_board_testcase/nightly/phoenix/hw"
    print(len(model_list))

    all_phoenis_models = os.listdir(org_dir)

    for model in model_list:
        if model in all_phoenis_models:
            org_model_dir = os.path.join(org_dir, model)
            det_model_dir = os.path.join(det_dir, model)
            # os.makedirs(det_model_dir)

            print("cp %s %s" % (org_model_dir, det_model_dir))
            shutil.copytree(org_model_dir, det_model_dir)


def make_nightly_json(model_list, from_json, to_json):
    node = platform.node()
    if node.startswith("xsj"):
        input_saved_path = "/proj/xsjhdstaff4/yanjunz/hugging_face_models"
    elif node.startswith("xco"):
        input_saved_path = "/proj/xcohdstaff5/huizhang/hugging_face_models"
    else:
        input_saved_path = ""

    with open(from_json, "r") as f:
        models = json.load(f)

    model_with_details_list = []
    for model in models:
        if model.get("id", "") not in model_list:
            continue
        model_dict = {}
        model_dict["id"] = model["id"]
        model_dict["onnx_model"] = model["onnx_model"]
        if "onnx_data" in model:
            model_dict["onnx_data"] = model["onnx_data"]
        input_path = os.path.join(input_saved_path, model["id"], "test_data_set_0")
        if os.path.exists(input_path):
            model_dict["input"] = input_path
        model_dict["hostname"] = model["hostname"]
        model_dict["md5sum"] = model["md5sum"]
        if model.get("golden", ""):
            model_dict["golden"] = model["golden"]

        model_with_details_list.append(model_dict)

    with open(to_json, "w") as f:
        json.dump(model_with_details_list, f, indent=4)


def make_modelzoo_json(modelzoo_path, json_file):
    res = []
    model_list = [
        x
        for x in os.listdir(modelzoo_path)
        if os.path.isdir(os.path.join(modelzoo_path, x))
    ]
    for model in model_list:
        onnx_models = [
            y
            for y in os.listdir(os.path.join(modelzoo_path, model))
            if y.endswith(".onnx")
        ]
        if len(onnx_models) != 1:
            print("ERROR: %s not HAS or ONLY HAS ONE onnx model!" % model)
            continue
        onnx_model = os.path.join(modelzoo_path, model, onnx_models[0])
        with open(onnx_model, "rb") as fb:
            md5 = hashlib.md5(fb.read()).hexdigest()
        has_onnx_data = os.path.isfile(os.path.join(onnx_model + ".data"))
        if has_onnx_data:
            model_info = {
                "id": model,
                "onnx_model": onnx_model,
                "onnx_data": onnx_model + ".data",
                "hostname": "xcdl190074.xilinx.com",
                "md5sum": md5,
            }
        else:
            model_info = {
                "id": model,
                "onnx_model": onnx_model,
                "hostname": "xcdl190074.xilinx.com",
                "md5sum": md5,
            }
        res.append(model_info)

        with open(json_file, "w") as file:
            file.write(json.dumps(res, indent=4))


if __name__ == "__main__":
    # make modelzoo json
    # modelzoo_path = '/proj/ipu_models/hug_fc/onnx_req_gpu/'
    modelzoo_path = (
        "/group/modelzoo/vai_q_onnx/P1_U8S8_quantized_models_aea0ea4_opset17"
    )
    make_modelzoo_json(modelzoo_path, "weekly_294_opset17_u8s8_p1.json")

    test_onnx_data_models = "resnetv2_152x2_bit.goog_in21k_ft_in1k ig_resnext101_32x32d resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k_384 resnetv2_50.a1h_in1k"

    nightly_48_p0_models = "adv_inception_v3 densenet121 efficientnet_el_pruned.in1k efficientnet_em.ra2_in1k efficientnet_es.ra_in1k efficientnet_lite0.ra_in1k ese_vovnet19b_dw fbnetv3_b.ra2_in1k gernet_l.idstcv_in1k gernet_s.idstcv_in1k ghostnet_100 gluon_inception_v3 gluon_resnet101_v1s gluon_resnet152_v1c gluon_resnet18_v1b gluon_resnet34_v1b gluon_resnet50_v1d gluon_xception65 hardcorenas_b hardcorenas_f inception_v3 inception_v4 lcnet_050.ra2_in1k lcnet_100.ra2_in1k mobilenetv2_050.lamb_in1k mobilenetv2_100.ra_in1k mobilenetv2_120d.ra_in1k mobilenetv3_large_100.miil_in21k_ft_in1k mobilenetv3_rw.rmsp_in1k mobilenetv3_small_050.lamb_in1k repvgg_b0.rvgg_in1k resnet101d resnet18 resnet200d resnet26t resnet34d spnasnet_100.rmsp_in1k swsl_resnet18 swsl_resnet50 tf_inception_v3 tv_densenet121 tv_resnet34 tv_resnet50 vgg13 vgg16_bn vgg19_bn xception xception41p"
    nightly_74_p1_models = "botnet26t_256 cs3darknet_focus_l cs3darknet_focus_m cs3edgenet_x cspdarknet53 densenetblur121d dla102 dla60_res2next dla60x_c dpn92 dpn98 eca_nfnet_l0.ra2_in1k eca_resnet33ts.ra2_in1k ecaresnet101d ecaresnet26t ecaresnet50d edgenext_base edgenext_small_rw edgenext_small edgenext_x_small edgenext_xx_small efficientnet_b0.ra_in1k efficientnet_b1.ft_in1k efficientnetv2_rw_t.ra2_in1k ens_adv_inception_resnet_v2 ese_vovnet39b fbnetv3_g.ra2_in1k gcresnet50t.ra2_in1k gluon_resnext101_64x4d gluon_seresnext101_32x4d gmlp_s16_224.ra3_in1k hrnet_w18_small_v2 hrnet_w18_small ig_resnext101_32x16d lambda_resnet50ts legacy_seresnet101 legacy_seresnext26_32x4d mixer_b16_224.miil_in21k_ft_in1k mixnet_m.ft_in1k nf_regnet_b1.ra2_in1k nf_resnet50.ra2_in1k regnetv_040.ra3_in1k regnety_002.pycls_in1k regnetz_c16.ra3_in1k repvgg_a2.rvgg_in1k res2net101_26w_4s res2net50_14w_8s res2net50_26w_4s resmlp_12_224.fb_distilled_in1k resmlp_12_224.fb_in1k resnest14d resnest50d_4s2x40d resnet33ts.ra2_in1k resnet61q.ra2_in1k resnetv2_50x1_bit.goog_in21k_ft_in1k resnext101_64x4d resnext26ts.ra2_in1k resnext50d_32x4d rexnet_130.nav_in1k sebotnet33ts_256 selecsls42b selecsls60 semnasnet_075.rmsp_in1k semnasnet_100.rmsp_in1k seresnet33ts.ra2_in1k seresnet50 skresnet34 ssl_resnext50_32x4d swsl_resnext101_32x4d tf_efficientnet_lite4.in1k tf_mobilenetv3_small_100.in1k tinynet_d.in1k tv_resnext50_32x4d xception41"
    nightly_85_p1_models = "resnetv2_50x1_bit.goog_in21k_ft_in1k resnext101_64x4d resnext26ts.ra2_in1k resnext50d_32x4d rexnet_130.nav_in1k resnest14d resnest50d_4s2x40d resnet33ts.ra2_in1k mixnet_m.ft_in1k nf_regnet_b1.ra2_in1k nf_resnet50.ra2_in1k regnetv_040.ra3_in1k regnetv_064.ra3_in1k regnetz_c16.ra3_in1k repvgg_a2.rvgg_in1k ssl_resnext50_32x4d swsl_resnext101_32x4d tf_efficientnet_el.in1k tf_efficientnet_lite1.in1k tf_efficientnet_lite4.in1k tf_efficientnetv2_b0.in1k legacy_seresnet101 legacy_seresnext26_32x4d mixer_b16_224.miil_in21k_ft_in1k tf_mobilenetv3_small_100.in1k tinynet_d.in1k xception41 resnet61q.ra2_in1k resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k_384 res2net101_26w_4s res2net50_14w_8s res2net50_26w_4s res2net50_26w_8s resmlp_12_224.fb_distilled_in1k resmlp_12_224.fb_in1k dpn92 dpn98 eca_nfnet_l0.ra2_in1k eca_nfnet_l1.ra2_in1k eca_resnet33ts.ra2_in1k ecaresnet101d ecaresnet26t ecaresnet50d botnet26t_256 cs3darknet_focus_l cs3darknet_focus_m cs3edgenet_x cspdarknet53 gluon_resnext101_64x4d gluon_seresnext101_32x4d gmixer_24_224.ra3_in1k gmlp_s16_224.ra3_in1k hrnet_w18_small hrnet_w18_small_v2 ig_resnext101_32x16d lambda_resnet50ts sebotnet33ts_256 selecsls42b selecsls60 semnasnet_075.rmsp_in1k semnasnet_100.rmsp_in1k seresnet33ts.ra2_in1k seresnet50 dla60x_c dm_nfnet_f0.dm_in1k dpn68 efficientnetv2_rw_t.ra2_in1k ens_adv_inception_resnet_v2 ese_vovnet39b fbnetv3_g.ra2_in1k gcresnet50t.ra2_in1k skresnet34 edgenext_base edgenext_small edgenext_small_rw edgenext_x_small edgenext_xx_small efficientnet_b0.ra_in1k efficientnet_b1.ft_in1k efficientnet_b4.ra2_in1k efficientnet_b5.in12k_ft_in1k densenetblur121d dla102 dla60_res2next regnety_002.pycls_in1k"
    weekly_296_s1_list = "regnety_640.seer_ft_in1k regnetz_e8.ra3_in1k eca_nfnet_l2.ra3_in1k legacy_senet154 resnext101_64x4d regnetz_c16_evos.ch_in1k regnetx_320.tv2_in1k regnetx_320.pycls_in1k regnetz_d8.ra3_in1k regnety_160.tv2_in1k regnetz_d32.ra3_in1k regnety_080.ra3_in1k hrnet_w64 gc_efficientnetv2_rw_t.agc_in1k legacy_seresnext101_32x4d jx_nest_base eca_nfnet_l1.ra2_in1k efficientnet_b5.in12k_ft_in1k regnetx_120.pycls_in1k focalnet_tiny_lrf.ms_in1k resnetv2_101x1_bit.goog_in21k_ft_in1k hrnet_w32 regnetx_080.pycls_in1k focalnet_tiny_srf.ms_in1k resnext50d_32x4d rexnet_100.nav_in1k rexnet_130.nav_in1k rexnet_150.nav_in1k rexnet_200.nav_in1k rexnet_300.nav_in1k rexnetr_200.sw_in12k_ft_in1k rexnetr_300.sw_in12k_ft_in1k sebotnet33ts_256 selecsls42b selecsls60 selecsls60b semnasnet_075.rmsp_in1k semnasnet_100.rmsp_in1k sequencer2d_s seresnet33ts.ra2_in1k seresnet50 seresnext26d_32x4d seresnext26t_32x4d seresnext26ts.ch_in1k seresnext50_32x4d skresnet18 skresnet34 skresnext50_32x4d ssl_resnext50_32x4d swsl_resnext50_32x4d tf_efficientnet_el.in1k tf_efficientnet_em.in1k tf_efficientnet_es.in1k tf_efficientnet_lite0.in1k tf_efficientnet_lite1.in1k tf_efficientnet_lite2.in1k tf_efficientnet_lite3.in1k tf_efficientnet_lite4.in1k tf_efficientnetv2_b0.in1k tf_efficientnetv2_b1.in1k tf_efficientnetv2_b2.in1k tf_efficientnetv2_b3.in1k tf_efficientnetv2_b3.in21k_ft_in1k tf_mobilenetv3_large_minimal_100.in1k tf_mobilenetv3_small_075.in1k tf_mobilenetv3_small_100.in1k tf_mobilenetv3_small_minimal_100.in1k tinynet_a.in1k tinynet_d.in1k tv_resnext50_32x4d wide_resnet50_2 xception41 xception65 xception71"
    weekly_296_s2_list = "ssl_resnext101_32x16d seresnextaa101d_32x8d resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k ig_resnext101_32x8d resnetv2_152x2_bit.goog_in21k_ft_in1k gluon_seresnext101_64x4d regnety_320.swag_lc_in1k regnety_320.pycls_in1k regnety_320.tv2_in1k dm_nfnet_f1.dm_in1k regnety_160.swag_lc_in1k regnety_160.pycls_in1k resnetv2_50d_evos.ah_in1k gluon_resnext101_32x4d regnety_080.pycls_in1k regnety_080_tv.tv2_in1k sequencer2d_m repvgg_b3.rvgg_in1k seresnet152d regnetx_080.tv2_in1k regnety_006.pycls_in1k regnety_008.pycls_in1k regnety_008_tv.tv2_in1k regnety_016.pycls_in1k regnety_016.tv2_in1k regnety_032.pycls_in1k regnety_032.ra_in1k regnety_032.tv2_in1k regnety_040.pycls_in1k regnety_040.ra3_in1k regnety_064.pycls_in1k regnetz_b16.ra3_in1k repvgg_a2.rvgg_in1k repvgg_b1.rvgg_in1k repvgg_b1g4.rvgg_in1k repvgg_b2.rvgg_in1k repvgg_b2g4.rvgg_in1k res2net101_26w_4s res2net50_14w_8s res2net50_26w_4s res2net50_26w_6s res2net50_26w_8s res2next50 resmlp_12_224.fb_distilled_in1k resmlp_12_224.fb_in1k resmlp_24_224.fb_distilled_in1k resmlp_24_224.fb_in1k resmlp_36_224.fb_distilled_in1k resmlp_36_224.fb_in1k resmlp_big_24_224.fb_distilled_in1k resnest14d resnest26d resnest50d resnest50d_1s4x24d resnest50d_4s2x40d resnet32ts.ra2_in1k resnet33ts.ra2_in1k resnet50_gn resnet51q.ra2_in1k resnet61q.ra2_in1k resnetblur50 resnetrs101 resnetrs50 resnetv2_101.a1h_in1k resnetv2_50.a1h_in1k resnetv2_50d_gn.ah_in1k resnetv2_50x1_bit.goog_distilled_in1k resnetv2_50x1_bit.goog_in21k_ft_in1k resnext26ts.ra2_in1k resnext50_32x4d dla34 dla46_c dla46x_c dla60x_c"
    weekly_296_s3_list = "swsl_resnext101_32x16d regnety_160.swag_ft_in1k regnetz_d8_evos.ch_in1k regnety_160.sw_in12k_ft_in1k resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k_384 regnety_160.lion_in12k_ft_in1k regnetz_040_h.ra3_in1k swsl_resnext101_32x8d regnetz_040.ra3_in1k dpn107 dpn131 efficientnetv2_rw_m.agc_in1k dla102x2 regnety_120.pycls_in1k jx_nest_small regnetv_064.ra3_in1k resnetrs152 mixer_l16_224.goog_in21k_ft_in1k hrnet_w44 hrnet_w48 resnetv2_50x3_bit.goog_in21k_ft_in1k densenet161 repvgg_b3g4.rvgg_in1k hrnet_w30 efficientnetv2_rw_t.ra2_in1k ens_adv_inception_resnet_v2 ese_vovnet39b fbnetv3_g.ra2_in1k gcresnet33ts.ra2_in1k gcresnet50t.ra2_in1k gcresnext26ts.ch_in1k gcresnext50ts.ch_in1k gluon_resnext50_32x4d gluon_seresnext50_32x4d gmixer_24_224.ra3_in1k gmlp_s16_224.ra3_in1k hrnet_w18_small hrnet_w18_small_v2 inception_resnet_v2 jx_nest_tiny lambda_resnet26rpt_256 lambda_resnet26t lambda_resnet50ts legacy_seresnet101 legacy_seresnet18 legacy_seresnet34 legacy_seresnet50 legacy_seresnext26_32x4d legacy_seresnext50_32x4d mixer_b16_224.goog_in21k_ft_in1k mixer_b16_224.miil_in21k_ft_in1k mixnet_l.ft_in1k mixnet_m.ft_in1k mixnet_s.ft_in1k mixnet_xl.ra_in1k mnasnet_small.lamb_in1k nf_regnet_b1.ra2_in1k nf_resnet50.ra2_in1k nfnet_l0.ra2_in1k regnetx_002.pycls_in1k regnetx_004.pycls_in1k regnetx_004_tv.tv2_in1k regnetx_006.pycls_in1k regnetx_008.pycls_in1k regnetx_008.tv2_in1k regnetx_016.pycls_in1k regnetx_016.tv2_in1k regnetx_032.pycls_in1k regnetx_032.tv2_in1k regnetx_040.pycls_in1k regnetx_064.pycls_in1k regnety_002.pycls_in1k regnety_004.pycls_in1k regnety_004.tv2_in1k"
    weekly_296_s4_list = "regnety_320.swag_ft_in1k regnety_320.seer_ft_in1k seresnext101_32x8d seresnext101d_32x8d gluon_senet154 resnext101_32x8d ig_resnext101_32x32d ig_resnext101_32x16d ssl_resnext101_32x8d gluon_resnext101_64x4d resnetrs200 regnety_120.sw_in12k_ft_in1k tnt_s_patch16_224 regnetx_160.pycls_in1k regnetx_160.tv2_in1k gluon_seresnext101_32x4d legacy_seresnet152 wide_resnet101_2 dpn98 regnetv_040.ra3_in1k regnetz_c16.ra3_in1k hrnet_w40 resnest101e swsl_resnext101_32x4d ssl_resnext101_32x4d hrnet_w18 regnety_064.ra3_in1k ecaresnet101d botnet26t_256 cs3darknet_focus_l cs3darknet_focus_m cs3darknet_l cs3darknet_m cs3darknet_x cs3edgenet_x cs3se_edgenet_x cs3sedarknet_l cs3sedarknet_x cspdarknet53 cspresnet50 cspresnext50 densenetblur121d dla102 dla102x dla169 dla60 dla60_res2net dla60_res2next dla60x dm_nfnet_f0.dm_in1k dpn68 dpn68b dpn92 eca_botnext26ts_256 eca_nfnet_l0.ra2_in1k eca_resnet33ts.ra2_in1k eca_resnext26ts.ch_in1k ecaresnet101d_pruned ecaresnet26t ecaresnet50d ecaresnet50d_pruned ecaresnet50t ecaresnetlight edgenext_base edgenext_small edgenext_small_rw edgenext_x_small edgenext_xx_small efficientnet_b0.ra_in1k efficientnet_b1.ft_in1k efficientnet_b2.ra_in1k efficientnet_b3.ra2_in1k efficientnet_b4.ra2_in1k efficientnetv2_rw_s.ra2_in1k"
    weekly_96_p0_list = "resnet10t mobilenetv2_050.lamb_in1k resnet14t ssl_resnet18 gluon_resnet18_v1b resnet18 mobilenetv2_100.ra_in1k efficientnet_lite0.ra_in1k lcnet_050.ra2_in1k resnet18d mnasnet_100.rmsp_in1k lcnet_075.ra2_in1k efficientnet_es.ra_in1k lcnet_100.ra2_in1k efficientnet_es_pruned.in1k ese_vovnet19b_dw swsl_resnet18 mobilenetv2_140.ra_in1k gernet_s.idstcv_in1k spnasnet_100.rmsp_in1k resnet26 resnet26d mobilenetv2_110d.ra_in1k resnet26t fbnetc_100.rmsp_in1k efficientnet_em.ra2_in1k hardcorenas_a mobilenetv3_small_050.lamb_in1k mobilenetv3_small_075.lamb_in1k mobilenetv3_small_100.lamb_in1k resnet34 gluon_resnet34_v1b tv_resnet34 resnet34d mobilenetv2_120d.ra_in1k efficientnet_el_pruned.in1k gernet_m.idstcv_in1k efficientnet_el.ra_in1k hardcorenas_b mobilenetv3_large_100.miil_in21k_ft_in1k resnet50 mobilenetv3_large_100.ra_in1k ssl_resnet50 swsl_resnet50 gluon_resnet50_v1b gluon_resnet50_v1c gluon_resnet50_v1s mobilenetv3_rw.rmsp_in1k gluon_resnet50_v1d tv_resnet50 resnet50d resnetaa50 hardcorenas_c gernet_l.idstcv_in1k xception res2net50_48w_2s repvgg_b0.rvgg_in1k gluon_inception_v3 inception_v3 tf_inception_v3 adv_inception_v3 xception41p hardcorenas_e hardcorenas_f hardcorenas_d resnet101 gluon_resnet101_v1c resnet101d gluon_resnet101_v1b tv_resnet101 gluon_resnet101_v1s gluon_resnet101_v1d ghostnet_100 gluon_xception65 vgg13 vgg11_bn vgg13_bn vgg11 xception65p vgg16 inception_v4 vgg16_bn vgg19 vgg19_bn resnet152 resnet152d tv_resnet152 gluon_resnet152_v1c gluon_resnet152_v1b gluon_resnet152_v1s gluon_resnet152_v1d resnet200d tv_densenet121 densenet121 densenet169 fbnetv3_b.ra2_in1k"
    weekly_296_p1_list = [x for x in weekly_296_s1_list.split(" ") if x]
    weekly_296_p1_list += [x for x in weekly_296_s2_list.split(" ") if x]
    weekly_296_p1_list += [x for x in weekly_296_s3_list.split(" ") if x]
    weekly_296_p1_list += [x for x in weekly_296_s4_list.split(" ") if x]
    # print(len(weekly_296_list))
    # split_296_models([x for x in nightly_74_p1_models.split(" ") if x], weekly_296_list)

    # split_296_models([x for x in nightly_48_p0_models.split(" ") if x], [x for x in weekly_96_p0_list.split(" ") if x])

    # weekly_296_p1_listleft_list = [x for x in weekly_296_p1_list if x and x not in nightly_74_p1_models.split(" ")]
    # nighlty_left_list = [x for x in weekly_96_p0_list.split(" ") if x and x not in nightly_48_p0_models.split(" ")]
    # print(nighlty_left_list)

    # make_nightly_json(nighlty_left_list, "weekly_96_p0.json", 'weekly_96_p0_left.json')
    # make_nightly_json([x for x in weekly_296_s2_list.split(" ") if x], "weekly_296_p1.json", 'weekly_296_s2_p1.json')
    # make_nightly_json([x for x in weekly_296_s3_list.split(" ") if x], "weekly_296_p1.json", 'weekly_296_s3_p1.json')
    # make_nightly_json([x for x in weekly_296_s4_list.split(" ") if x], "weekly_296_p1.json", 'weekly_296_s4_p1.json')

    # make_nightly_json(
    #     [x for x in test_onnx_data_models.split(" ") if x],
    #     "test_onnx_data.json",
    #     "nightly_test_onnx_data.json",
    # )
    make_nightly_json(
        [x for x in nightly_85_p1_models.split(" ") if x],
        "weekly_294_opset17_u8s8_p1.json",
        "nightly_85_opset17_u8s8_p1.json",
    )

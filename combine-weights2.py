'''
For combining animeinterp+gma with gma-mixed.pth
'''

import torch

# from models.AnimeInterp import AnimeInterp

# model = AnimeInterp().cuda()
# print(model)

animeinterp = torch.load('checkpoints/animeinterp+gma.pth')
d1 = animeinterp['model_state_dict']
d2 = torch.load('checkpoints/gma-mixed.pth')['state_dict']

# print(d1.keys())
# print(d2.keys())
# for k in d2.keys():
#     print(k)

d1["module.flownet.fnet.conv1.weight"] = d2["encoder.conv1.weight"]
d1["module.flownet.fnet.conv1.bias"] = d2["encoder.conv1.bias"]
d1["module.flownet.fnet.conv2.weight"] = d2["encoder.conv2.weight"]
d1["module.flownet.fnet.conv2.bias"] = d2["encoder.conv2.bias"]
d1["module.flownet.fnet.layer1.0.conv1.weight"] = d2["encoder.res_layer1.0.conv1.weight"]
d1["module.flownet.fnet.layer1.0.conv1.bias"] = d2["encoder.res_layer1.0.conv1.bias"]
d1["module.flownet.fnet.layer1.0.conv2.weight"] = d2["encoder.res_layer1.0.conv2.weight"]
d1["module.flownet.fnet.layer1.0.conv2.bias"] = d2["encoder.res_layer1.0.conv2.bias"]
d1["module.flownet.fnet.layer1.1.conv1.weight"] = d2["encoder.res_layer1.1.conv1.weight"]
d1["module.flownet.fnet.layer1.1.conv1.bias"] = d2["encoder.res_layer1.1.conv1.bias"]
d1["module.flownet.fnet.layer1.1.conv2.weight"] = d2["encoder.res_layer1.1.conv2.weight"]
d1["module.flownet.fnet.layer1.1.conv2.bias"] = d2["encoder.res_layer1.1.conv2.bias"]
d1["module.flownet.fnet.layer2.0.conv1.weight"] = d2["encoder.res_layer2.0.conv1.weight"]
d1["module.flownet.fnet.layer2.0.conv1.bias"] = d2["encoder.res_layer2.0.conv1.bias"]
d1["module.flownet.fnet.layer2.0.conv2.weight"] = d2["encoder.res_layer2.0.conv2.weight"]
d1["module.flownet.fnet.layer2.0.conv2.bias"] = d2["encoder.res_layer2.0.conv2.bias"]
d1["module.flownet.fnet.layer2.0.downsample.0.weight"] = d2["encoder.res_layer2.0.downsample.0.weight"]
d1["module.flownet.fnet.layer2.0.downsample.0.bias"] = d2["encoder.res_layer2.0.downsample.0.bias"]
d1["module.flownet.fnet.layer2.1.conv1.weight"] = d2["encoder.res_layer2.1.conv1.weight"]
d1["module.flownet.fnet.layer2.1.conv1.bias"] = d2["encoder.res_layer2.1.conv1.bias"]
d1["module.flownet.fnet.layer2.1.conv2.weight"] = d2["encoder.res_layer2.1.conv2.weight"]
d1["module.flownet.fnet.layer2.1.conv2.bias"] = d2["encoder.res_layer2.1.conv2.bias"]
d1["module.flownet.fnet.layer3.0.conv1.weight"] = d2["encoder.res_layer3.0.conv1.weight"]
d1["module.flownet.fnet.layer3.0.conv1.bias"] = d2["encoder.res_layer3.0.conv1.bias"]
d1["module.flownet.fnet.layer3.0.conv2.weight"] = d2["encoder.res_layer3.0.conv2.weight"]
d1["module.flownet.fnet.layer3.0.conv2.bias"] = d2["encoder.res_layer3.0.conv2.bias"]
d1["module.flownet.fnet.layer3.0.downsample.0.weight"] = d2["encoder.res_layer3.0.downsample.0.weight"]
d1["module.flownet.fnet.layer3.0.downsample.0.bias"] = d2["encoder.res_layer3.0.downsample.0.bias"]
d1["module.flownet.fnet.layer3.1.conv1.weight"] = d2["encoder.res_layer3.1.conv1.weight"]
d1["module.flownet.fnet.layer3.1.conv1.bias"] = d2["encoder.res_layer3.1.conv1.bias"]
d1["module.flownet.fnet.layer3.1.conv2.weight"] = d2["encoder.res_layer3.1.conv2.weight"]
d1["module.flownet.fnet.layer3.1.conv2.bias"] = d2["encoder.res_layer3.1.conv2.bias"]
d1["module.flownet.update_block.encoder.convc1.weight"] = d2["decoder.encoder.corr_net.0.conv.weight"]
d1["module.flownet.update_block.encoder.convc1.bias"] = d2["decoder.encoder.corr_net.0.conv.bias"]
d1["module.flownet.update_block.encoder.convc2.weight"] = d2["decoder.encoder.corr_net.1.conv.weight"]
d1["module.flownet.update_block.encoder.convc2.bias"] = d2["decoder.encoder.corr_net.1.conv.bias"]
d1["module.flownet.update_block.encoder.convf1.weight"] = d2["decoder.encoder.flow_net.0.conv.weight"]
d1["module.flownet.update_block.encoder.convf1.bias"] = d2["decoder.encoder.flow_net.0.conv.bias"]
d1["module.flownet.update_block.encoder.convf2.weight"] = d2["decoder.encoder.flow_net.1.conv.weight"]
d1["module.flownet.update_block.encoder.convf2.bias"] = d2["decoder.encoder.flow_net.1.conv.bias"]
d1["module.flownet.update_block.encoder.conv.weight"] = d2["decoder.encoder.out_net.0.conv.weight"]
d1["module.flownet.update_block.encoder.conv.bias"] = d2["decoder.encoder.out_net.0.conv.bias"]
d1["module.flownet.update_block.gru.convz1.weight"] = d2["decoder.gru.conv_z.0.conv.weight"]
d1["module.flownet.update_block.gru.convz1.bias"] = d2["decoder.gru.conv_z.0.conv.bias"]
d1["module.flownet.update_block.gru.convz2.weight"] = d2["decoder.gru.conv_z.1.conv.weight"]
d1["module.flownet.update_block.gru.convz2.bias"] = d2["decoder.gru.conv_z.1.conv.bias"]
d1["module.flownet.update_block.gru.convr1.weight"] = d2["decoder.gru.conv_r.0.conv.weight"]
d1["module.flownet.update_block.gru.convr1.bias"] = d2["decoder.gru.conv_r.0.conv.bias"]
d1["module.flownet.update_block.gru.convr2.weight"] = d2["decoder.gru.conv_r.1.conv.weight"]
d1["module.flownet.update_block.gru.convr2.bias"] = d2["decoder.gru.conv_r.1.conv.bias"]
d1["module.flownet.update_block.gru.convq1.weight"] = d2["decoder.gru.conv_q.0.conv.weight"]
d1["module.flownet.update_block.gru.convq1.bias"] = d2["decoder.gru.conv_q.0.conv.bias"]
d1["module.flownet.update_block.gru.convq2.weight"] = d2["decoder.gru.conv_q.1.conv.weight"]
d1["module.flownet.update_block.gru.convq2.bias"] = d2["decoder.gru.conv_q.1.conv.bias"]
d1["module.flownet.update_block.flow_head.conv1.weight"] = d2["decoder.flow_pred.layers.0.conv.weight"]
d1["module.flownet.update_block.flow_head.conv1.bias"] = d2["decoder.flow_pred.layers.0.conv.bias"]
d1["module.flownet.update_block.flow_head.conv2.weight"] = d2["decoder.flow_pred.predict_layer.weight"]
d1["module.flownet.update_block.flow_head.conv2.bias"] = d2["decoder.flow_pred.predict_layer.bias"]
d1["module.flownet.update_block.mask.0.weight"] = d2["decoder.mask_pred.layers.0.conv.weight"]
d1["module.flownet.update_block.mask.0.bias"] = d2["decoder.mask_pred.layers.0.conv.bias"]
d1["module.flownet.update_block.mask.2.weight"] = d2["decoder.mask_pred.predict_layer.weight"]
d1["module.flownet.update_block.mask.2.bias"] = d2["decoder.mask_pred.predict_layer.bias"]
d1["module.flownet.att.to_qk.weight"] = d2["decoder.attn.to_qk.weight"]
d1["module.flownet.att.pos_emb.rel_ind"] = d2["decoder.attn.pos_emb.rel_ind"]
d1["module.flownet.att.pos_emb.rel_height.weight"] = d2["decoder.attn.pos_emb.rel_height.weight"]
d1["module.flownet.att.pos_emb.rel_width.weight"] = d2["decoder.attn.pos_emb.rel_width.weight"]
d1["module.flownet.update_block.aggregator.gamma"] = d2["decoder.aggregator.gamma"]
d1["module.flownet.update_block.aggregator.to_v.weight"] = d2["decoder.aggregator.to_v.weight"]
d1["module.flownet.cnet.conv1.weight"] = d2["context.conv1.weight"]
d1["module.flownet.cnet.conv1.bias"] = d2["context.conv1.bias"]
d1["module.flownet.cnet.norm1.weight"] = d2["context.bn1.weight"]
d1["module.flownet.cnet.norm1.bias"] = d2["context.bn1.bias"]
d1["module.flownet.cnet.norm1.running_mean"] = d2["context.bn1.running_mean"]
d1["module.flownet.cnet.norm1.running_var"] = d2["context.bn1.running_var"]
d1["module.flownet.cnet.norm1.num_batches_tracked"] = d2["context.bn1.num_batches_tracked"]
d1["module.flownet.cnet.layer1.0.conv1.weight"] = d2["context.res_layer1.0.conv1.weight"]
d1["module.flownet.cnet.layer1.0.conv1.bias"] = d2["context.res_layer1.0.conv1.bias"]
d1["module.flownet.cnet.layer1.0.norm1.weight"] = d2["context.res_layer1.0.bn1.weight"]
d1["module.flownet.cnet.layer1.0.norm1.bias"] = d2["context.res_layer1.0.bn1.bias"]
d1["module.flownet.cnet.layer1.0.norm1.running_mean"] = d2["context.res_layer1.0.bn1.running_mean"]
d1["module.flownet.cnet.layer1.0.norm1.running_var"] = d2["context.res_layer1.0.bn1.running_var"]
d1["module.flownet.cnet.layer1.0.norm1.num_batches_tracked"] = d2["context.res_layer1.0.bn1.num_batches_tracked"]
d1["module.flownet.cnet.layer1.0.conv2.weight"] = d2["context.res_layer1.0.conv2.weight"]
d1["module.flownet.cnet.layer1.0.conv2.bias"] = d2["context.res_layer1.0.conv2.bias"]
d1["module.flownet.cnet.layer1.0.norm2.weight"] = d2["context.res_layer1.0.bn2.weight"]
d1["module.flownet.cnet.layer1.0.norm2.bias"] = d2["context.res_layer1.0.bn2.bias"]
d1["module.flownet.cnet.layer1.0.norm2.running_mean"] = d2["context.res_layer1.0.bn2.running_mean"]
d1["module.flownet.cnet.layer1.0.norm2.running_var"] = d2["context.res_layer1.0.bn2.running_var"]
d1["module.flownet.cnet.layer1.0.norm2.num_batches_tracked"] = d2["context.res_layer1.0.bn2.num_batches_tracked"]
d1["module.flownet.cnet.layer1.1.conv1.weight"] = d2["context.res_layer1.1.conv1.weight"]
d1["module.flownet.cnet.layer1.1.conv1.bias"] = d2["context.res_layer1.1.conv1.bias"]
d1["module.flownet.cnet.layer1.1.norm1.weight"] = d2["context.res_layer1.1.bn1.weight"]
d1["module.flownet.cnet.layer1.1.norm1.bias"] = d2["context.res_layer1.1.bn1.bias"]
d1["module.flownet.cnet.layer1.1.norm1.running_mean"] = d2["context.res_layer1.1.bn1.running_mean"]
d1["module.flownet.cnet.layer1.1.norm1.running_var"] = d2["context.res_layer1.1.bn1.running_var"]
d1["module.flownet.cnet.layer1.1.norm1.num_batches_tracked"] = d2["context.res_layer1.1.bn1.num_batches_tracked"]
d1["module.flownet.cnet.layer1.1.conv2.weight"] = d2["context.res_layer1.1.conv2.weight"]
d1["module.flownet.cnet.layer1.1.conv2.bias"] = d2["context.res_layer1.1.conv2.bias"]
d1["module.flownet.cnet.layer1.1.norm2.weight"] = d2["context.res_layer1.1.bn2.weight"]
d1["module.flownet.cnet.layer1.1.norm2.bias"] = d2["context.res_layer1.1.bn2.bias"]
d1["module.flownet.cnet.layer1.1.norm2.running_mean"] = d2["context.res_layer1.1.bn2.running_mean"]
d1["module.flownet.cnet.layer1.1.norm2.running_var"] = d2["context.res_layer1.1.bn2.running_var"]
d1["module.flownet.cnet.layer1.1.norm2.num_batches_tracked"] = d2["context.res_layer1.1.bn2.num_batches_tracked"]
d1["module.flownet.cnet.conv2.weight"] = d2["context.conv2.weight"]
d1["module.flownet.cnet.conv2.bias"] = d2["context.conv2.bias"]
d1["module.flownet.cnet.layer2.0.conv1.weight"] = d2["context.res_layer2.0.conv1.weight"]
d1["module.flownet.cnet.layer2.0.conv1.bias"] = d2["context.res_layer2.0.conv1.bias"]
d1["module.flownet.cnet.layer2.0.norm1.weight"] = d2["context.res_layer2.0.bn1.weight"]
d1["module.flownet.cnet.layer2.0.norm1.bias"] = d2["context.res_layer2.0.bn1.bias"]
d1["module.flownet.cnet.layer2.0.norm1.running_mean"] = d2["context.res_layer2.0.bn1.running_mean"]
d1["module.flownet.cnet.layer2.0.norm1.running_var"] = d2["context.res_layer2.0.bn1.running_var"]
d1["module.flownet.cnet.layer2.0.norm1.num_batches_tracked"] = d2["context.res_layer2.0.bn1.num_batches_tracked"]
d1["module.flownet.cnet.layer2.0.conv2.weight"] = d2["context.res_layer2.0.conv2.weight"]
d1["module.flownet.cnet.layer2.0.conv2.bias"] = d2["context.res_layer2.0.conv2.bias"]
d1["module.flownet.cnet.layer2.0.norm2.weight"] = d2["context.res_layer2.0.bn2.weight"]
d1["module.flownet.cnet.layer2.0.norm2.bias"] = d2["context.res_layer2.0.bn2.bias"]
d1["module.flownet.cnet.layer2.0.norm2.running_mean"] = d2["context.res_layer2.0.bn2.running_mean"]
d1["module.flownet.cnet.layer2.0.norm2.running_var"] = d2["context.res_layer2.0.bn2.running_var"]
d1["module.flownet.cnet.layer2.0.norm2.num_batches_tracked"] = d2["context.res_layer2.0.bn2.num_batches_tracked"]
# norm3 missing!
d1["module.flownet.cnet.layer2.0.downsample.0.weight"] = d2["context.res_layer2.0.downsample.0.weight"]
d1["module.flownet.cnet.layer2.0.downsample.0.bias"] = d2["context.res_layer2.0.downsample.0.bias"]
d1["module.flownet.cnet.layer2.0.downsample.1.weight"] = d2["context.res_layer2.0.downsample.1.weight"]
d1["module.flownet.cnet.layer2.0.downsample.1.bias"] = d2["context.res_layer2.0.downsample.1.bias"]
d1["module.flownet.cnet.layer2.0.downsample.1.running_mean"] = d2["context.res_layer2.0.downsample.1.running_mean"]
d1["module.flownet.cnet.layer2.0.downsample.1.running_var"] = d2["context.res_layer2.0.downsample.1.running_var"]
d1["module.flownet.cnet.layer2.0.downsample.1.num_batches_tracked"] = d2["context.res_layer2.0.downsample.1.num_batches_tracked"]
d1["module.flownet.cnet.layer2.1.conv1.weight"] = d2["context.res_layer2.1.conv1.weight"]
d1["module.flownet.cnet.layer2.1.conv1.bias"] = d2["context.res_layer2.1.conv1.bias"]
d1["module.flownet.cnet.layer2.1.norm1.weight"] = d2["context.res_layer2.1.bn1.weight"]
d1["module.flownet.cnet.layer2.1.norm1.bias"] = d2["context.res_layer2.1.bn1.bias"]
d1["module.flownet.cnet.layer2.1.norm1.running_mean"] = d2["context.res_layer2.1.bn1.running_mean"]
d1["module.flownet.cnet.layer2.1.norm1.running_var"] = d2["context.res_layer2.1.bn1.running_var"]
d1["module.flownet.cnet.layer2.1.norm1.num_batches_tracked"] = d2["context.res_layer2.1.bn1.num_batches_tracked"]
d1["module.flownet.cnet.layer2.1.conv2.weight"] = d2["context.res_layer2.1.conv2.weight"]
d1["module.flownet.cnet.layer2.1.conv2.bias"] = d2["context.res_layer2.1.conv2.bias"]
d1["module.flownet.cnet.layer2.1.norm2.weight"] = d2["context.res_layer2.1.bn2.weight"]
d1["module.flownet.cnet.layer2.1.norm2.bias"] = d2["context.res_layer2.1.bn2.bias"]
d1["module.flownet.cnet.layer2.1.norm2.running_mean"] = d2["context.res_layer2.1.bn2.running_mean"]
d1["module.flownet.cnet.layer2.1.norm2.running_var"] = d2["context.res_layer2.1.bn2.running_var"]
d1["module.flownet.cnet.layer2.1.norm2.num_batches_tracked"] = d2["context.res_layer2.1.bn2.num_batches_tracked"]
d1["module.flownet.cnet.layer3.0.conv1.weight"] = d2["context.res_layer3.0.conv1.weight"]
d1["module.flownet.cnet.layer3.0.conv1.bias"] = d2["context.res_layer3.0.conv1.bias"]
d1["module.flownet.cnet.layer3.0.norm1.weight"] = d2["context.res_layer3.0.bn1.weight"]
d1["module.flownet.cnet.layer3.0.norm1.bias"] = d2["context.res_layer3.0.bn1.bias"]
d1["module.flownet.cnet.layer3.0.norm1.running_mean"] = d2["context.res_layer3.0.bn1.running_mean"]
d1["module.flownet.cnet.layer3.0.norm1.running_var"] = d2["context.res_layer3.0.bn1.running_var"]
d1["module.flownet.cnet.layer3.0.norm1.num_batches_tracked"] = d2["context.res_layer3.0.bn1.num_batches_tracked"]
d1["module.flownet.cnet.layer3.0.conv2.weight"] = d2["context.res_layer3.0.conv2.weight"]
d1["module.flownet.cnet.layer3.0.conv2.bias"] = d2["context.res_layer3.0.conv2.bias"]
d1["module.flownet.cnet.layer3.0.norm2.weight"] = d2["context.res_layer3.0.bn2.weight"]
d1["module.flownet.cnet.layer3.0.norm2.bias"] = d2["context.res_layer3.0.bn2.bias"]
d1["module.flownet.cnet.layer3.0.norm2.running_mean"] = d2["context.res_layer3.0.bn2.running_mean"]
d1["module.flownet.cnet.layer3.0.norm2.running_var"] = d2["context.res_layer3.0.bn2.running_var"]
d1["module.flownet.cnet.layer3.0.norm2.num_batches_tracked"] = d2["context.res_layer3.0.bn2.num_batches_tracked"]
d1["module.flownet.cnet.layer3.0.downsample.0.weight"] = d2["context.res_layer3.0.downsample.0.weight"]
d1["module.flownet.cnet.layer3.0.downsample.0.bias"] = d2["context.res_layer3.0.downsample.0.bias"]
d1["module.flownet.cnet.layer3.0.downsample.1.weight"] = d2["context.res_layer3.0.downsample.1.weight"]
d1["module.flownet.cnet.layer3.0.downsample.1.bias"] = d2["context.res_layer3.0.downsample.1.bias"]
d1["module.flownet.cnet.layer3.0.downsample.1.running_mean"] = d2["context.res_layer3.0.downsample.1.running_mean"]
d1["module.flownet.cnet.layer3.0.downsample.1.running_var"] = d2["context.res_layer3.0.downsample.1.running_var"]
d1["module.flownet.cnet.layer3.0.downsample.1.num_batches_tracked"] = d2["context.res_layer3.0.downsample.1.num_batches_tracked"]
# norm3 missing!
d1["module.flownet.cnet.layer3.1.conv1.weight"] = d2["context.res_layer3.1.conv1.weight"]
d1["module.flownet.cnet.layer3.1.conv1.bias"] = d2["context.res_layer3.1.conv1.bias"]
d1["module.flownet.cnet.layer3.1.norm1.weight"] = d2["context.res_layer3.1.bn1.weight"]
d1["module.flownet.cnet.layer3.1.norm1.bias"] = d2["context.res_layer3.1.bn1.bias"]
d1["module.flownet.cnet.layer3.1.norm1.running_mean"] = d2["context.res_layer3.1.bn1.running_mean"]
d1["module.flownet.cnet.layer3.1.norm1.running_var"] = d2["context.res_layer3.1.bn1.running_var"]
d1["module.flownet.cnet.layer3.1.norm1.num_batches_tracked"] = d2["context.res_layer3.1.bn1.num_batches_tracked"]
d1["module.flownet.cnet.layer3.1.conv2.weight"] = d2["context.res_layer3.1.conv2.weight"]
d1["module.flownet.cnet.layer3.1.conv2.bias"] = d2["context.res_layer3.1.conv2.bias"]
d1["module.flownet.cnet.layer3.1.norm2.weight"] = d2["context.res_layer3.1.bn2.weight"]
d1["module.flownet.cnet.layer3.1.norm2.bias"] = d2["context.res_layer3.1.bn2.bias"]
d1["module.flownet.cnet.layer3.1.norm2.running_mean"] = d2["context.res_layer3.1.bn2.running_mean"]
d1["module.flownet.cnet.layer3.1.norm2.running_var"] = d2["context.res_layer3.1.bn2.running_var"]
d1["module.flownet.cnet.layer3.1.norm2.num_batches_tracked"] = d2["context.res_layer3.1.bn2.num_batches_tracked"]



# print(animeinterp['model_state_dict'].keys())
torch.save(animeinterp, "animeinterp+gma mixed.pth")

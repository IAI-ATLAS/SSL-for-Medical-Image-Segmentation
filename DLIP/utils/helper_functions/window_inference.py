import slidingwindow as sw
import torch 

def window_inference(model, input_tensor, n_classes=1, window_size=512, boarder=12):
    windows = sw.generate(data=input_tensor.numpy(),
                        dimOrder=sw.DimOrder.ChannelHeightWidth,
                        maxWindowSize=window_size,
                        overlapPercent=0.1)

    prediction = torch.zeros((n_classes, input_tensor.shape[1],input_tensor.shape[2]))

    for window in windows:
        h = boarder if window.indices()[1].start > 0 else 0
        w = boarder if window.indices()[2].start > 0 else 0
        hmin = window.indices()[1].start + h
        wmin = window.indices()[2].start + w
        net_input_window = input_tensor[window.indices()].cuda()
        prediction_window = model(net_input_window.unsqueeze(0)).detach().cpu()
        # Combine the sliding windows
        prediction[:, hmin:window.indices()[1].stop, wmin:window.indices()[2].stop] = prediction_window[0,:, h:, w:]

    return prediction
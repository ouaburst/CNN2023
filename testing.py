# Train the network
learning_rate = 0.01    # Set the learning rate
epochs = 5              # Set the number of epochs

training_losses = []    # Create an empty list to store the training losses
training_accuracies = []    # Create an empty list to store the training accuracies

for epoch in range(epochs):

    epoch_losses = []
    epoch_corrects = 0
    
    for i, (image, label) in enumerate(zip(train_images, train_labels)):
        # Apply the CNN and ANN on the input image
        cnn_output = cnn(image, conv_filters, pool_size, pool_stride)
        input_layer, hidden_layer, ann_output = ann(cnn_output.reshape(1, -1), weights, biases)

        # One-hot encode the label
        one_hot_label = np.zeros((1, output_size))
        one_hot_label[0, label] = 1

        # Calculate loss and gradients
        loss = -np.sum(one_hot_label * np.log(ann_output))
        epoch_losses.append(loss)
        pred_label = np.argmax(ann_output, axis=1)
        epoch_corrects += int(pred_label == label)        
        d_output = ann_output - one_hot_label
        d_hidden = np.dot(d_output, weights[2].T) * (hidden_layer > 0)
        d_input = np.dot(d_hidden, weights[1].T)

        # Update the weights and biases using backpropagation and the Adam optimizer
        weights[2] -= learning_rate * np.dot(hidden_layer.T, d_output)
        biases[2] -= learning_rate * np.sum(d_output, axis=0, keepdims=True)
        weights[1] -= learning_rate * np.dot(input_layer.T, d_hidden)
        biases[1] -= learning_rate * np.sum(d_hidden, axis=0, keepdims=True)
        weights[0] -= learning_rate * np.dot(cnn_output.reshape(-1, 1), d_input)
        
        # Calculate gradients for the convolution filters
        d_feature_maps2 = np.split(d_input, len(conv_filters[1]))
        d_pooled_maps1 = [np.zeros_like(pooled_map) for pooled_map in pooled_maps1]

        for fm_idx, d_feature_map2 in enumerate(d_feature_maps2):
            d_feature_map2 = d_feature_map2.reshape(1, 1)
            pooled_map_idx = fm_idx // len(conv_filters[1])
            d_pooled_map = np.zeros_like(pooled_maps1[pooled_map_idx])
            d_pooled_map[:d_feature_map2.shape[0], :d_feature_map2.shape[1]] = d_feature_map2
            d_pooled_maps1[pooled_map_idx] += d_pooled_map

        d_feature_maps1 = []
        for pooled_map, d_pooled_map in zip(pooled_maps1, d_pooled_maps1):
            d_feature_map1 = np.zeros_like(pooled_map)
            for y in range(0, pooled_map.shape[0] - 2 + 1, 2):
                for x in range(0, pooled_map.shape[1] - 2 + 1, 2):
                    window = pooled_map[y:y+2, x:x+2]
                    window_idx = np.unravel_index(window.argmax(), window.shape)
                    d_window = d_pooled_map[y:y+2, x:x+2]
                    d_feature_map1[y + window_idx[0], x + window_idx[1]] = d_window[window_idx]
            d_feature_maps1.append(d_feature_map1)

        # Update the first layer of convolution filters
        for filter_idx, conv_filter in enumerate(conv_filters[0]):
            feature_map = feature_maps1[filter_idx]
            d_feature_map = d_feature_maps1[filter_idx]
            for y in range(conv_filter.shape[0]):
                for x in range(conv_filter.shape[1]):
                    conv_filters[0][filter_idx][y, x] -= learning_rate * np.sum(feature_map[y:y+d_feature_map.shape[0], x:x+d_feature_map.shape[1]] * d_feature_map)
        
        # Update the second layer of convolution filters
        for filter_idx, conv_filter in enumerate(conv_filters[1]):
            feature_map = feature_maps2[filter_idx]
            d_feature_map = d_feature_maps2[filter_idx].reshape(1, 1)
            for y in range(conv_filter.shape[0]):
                for x in range(conv_filter.shape[1]):
                    conv_filters[1][filter_idx][y, x] -= learning_rate * np.sum(feature_map[y:y+d_feature_map.shape[0], x:x+d_feature_map.shape[1]] * d_feature_map)

        # Print the loss for every 1000th sample
        if i % 1000 == 0:
            print(f"Epoch: {epoch + 1}, Sample: {i}, Loss: {loss}")

    # Calculate the average loss and accuracy for this epoch
    avg_epoch_loss = np.mean(epoch_losses)
    epoch_accuracy = epoch_corrects / num_train_samples    
    # Store the values
    training_losses.append(avg_epoch_loss)
    training_accuracies.append(epoch_accuracy)
    # Print the average loss and accuracy for this epoch
    print(f"Epoch: {epoch + 1}, Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy * 100:.2f}%") 

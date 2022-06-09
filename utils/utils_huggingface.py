import torch.nn as nn

def preprocess_layers(args, model, model_type="distilbert-base-uncased"):

    if model_type=="distilbert-base-uncased":
        if args.train_one_layer:
            # Use this when fine-tuning only the last layer
            for name, param in model.distilbert.named_parameters():
                param.requires_grad = False
            for name, param in model.distilbert.transformer.layer[5].named_parameters():
                print(f"Only allow the training in layer 5-{name}")
                param.requires_grad = True
            
        if args.randomize_layers:
            # Use this when initializing the weights of the last layer
            for layer_id in range(args.randomize_layers_num):
                for name, param in model.distilbert.transformer.layer[5-layer_id].named_modules():
                    if isinstance(param, (nn.Linear, nn.Embedding, nn.LayerNorm)):
                        model._init_weights(param)
                        print(f"Randomizing the weights of layer {5-layer_id}-{name}")
    
    elif model_type == "distilroberta-base":
        if args.train_one_layer:
            # Use this when fine-tuning only the last layer
            raise NameError
            
        if args.randomize_layers:
            # Use this when initializing the weights of the last several layers
            for name, param in model.lm_head.named_modules():
                    if isinstance(param, (nn.Linear, nn.Embedding, nn.LayerNorm)):
                        model._init_weights(param)
                        print(f"Randomizing the weights of layer {name}")

            for layer_id in range(args.randomize_layers_num):
                for name, param in model.roberta.encoder.layer[5-layer_id].named_modules():
                    if isinstance(param, (nn.Linear, nn.Embedding, nn.LayerNorm)):
                        model._init_weights(param)
                        print(f"Randomizing the weights of layer {5-layer_id}-{name}")

    else:
        # It looks like randomizing layers is not the right way to go
        # We thus do not pursue this direction further
        if args.train_one_layer:
            raise NameError
        
        if args.randomize_layers:
            raise NameError
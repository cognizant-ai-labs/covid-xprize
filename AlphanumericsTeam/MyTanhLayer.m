classdef MyTanhLayer < nnet.layer.Layer
    % A tanh NN layer with learnable factor.
    % Reza Sameni
    % Dec 2020
    
    properties (Learnable)
        % Layer learnable parameters.
        
        % Expopent coefficient.
        Alpha
    end
    
    methods
        function layer = MyTanhLayer(numChannels, name, initial_param_std)
            % layer = MyTanhLayer(numChannels, name) creates a tanh layer
            % with numChannels channels and specifies the layer name.
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "Tanh with " + numChannels + " channels";
            
            % Initialize scaling coefficient.
            layer.Alpha = initial_param_std * randn(numChannels);
        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            
            Z = layer.Alpha .* tanh(X ./ layer.Alpha);
        end
    end
end
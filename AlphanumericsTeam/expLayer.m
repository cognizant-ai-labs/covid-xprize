classdef expLayer < nnet.layer.Layer
    % An exponential NN layer.
    % Reza Sameni
    % Dec 2020
    
    properties (Learnable)
        % Layer learnable parameters.
        
        % Expopent coefficient.
        Alpha
    end
    
    methods
        function layer = expLayer(numChannels, name)
            % layer = expLayer(numChannels, name) creates an Exponential layer
            % with numChannels channels and specifies the layer name.
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "Exp with " + numChannels + " channels";
            
            % Initialize scaling coefficient.
            layer.Alpha = rand([1 1 numChannels]);
        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            
            Z = exp(layer.Alpha .* X);
        end
    end
end
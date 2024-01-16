classdef Normalizer
    properties
        average;
        variance;
        dim;
        eps;
        divider;
    end
    methods
        function obj = Normalizer(dim, data, e)
            if nargin < 3
                e = 1e-5;
            end
            obj.dim = dim;
            obj.eps = e;
            obj.average = mean(data, dim);
            obj.variance = var(data, 0, dim);
            obj.divider = sqrt(obj.variance + obj.eps);
        end
        function obj = fit(obj, data)
            obj.average = mean(data, obj.dim);
           obj.variance = var(data, 0, obj.dim);
           obj.divider = sqrt(obj.variance + obj.eps);
        end
        function B = transform(obj, A)
            B = (A - obj.average) ./ obj.divider;
        end
        function obj = SetEps(obj, e)
            obj.eps = e;
        end
    end
end
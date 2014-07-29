function [value index distances] = findTwoNearest(Input,nodes, M) %#codegen

NumOfNodes = size(nodes,2);
distances = zeros(1,NumOfNodes);
value = [0 0];
index = [0 0];

for i=1:NumOfNodes
    distances(i) = (Input - nodes(:,i))'*M*(Input - nodes(:,i));
end

[sdistances idx] = sort(distances);

value(1) = sdistances(1);
value(2) = sdistances(2);
index(1) = idx(1);
index(2) = idx(2);

 
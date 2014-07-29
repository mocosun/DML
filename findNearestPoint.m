function [pointsFound] = findNearestPoint(points, nodeToFind, numOfNearestPoint, M)
NumOfPoints = size(points,2);

distances = zeros(1,NumOfPoints);
for i=1:NumOfPoints
    distances(i) = (nodeToFind - points(:,i))'*M*(nodeToFind - points(:,i));
end

[sdistances indices] = sort(distances); 

pointsFound = points(:,indices(:,1:numOfNearestPoint));



end
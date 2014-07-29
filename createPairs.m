function [idxa, idxb] = createPairs(sampleData, SampleCountEachPerson)
%idxa: 索引sampleData，前一半是相似图片对的第一张图片， 后一半是不相似图片对的第一张图片
%idxb: 索引sampleData，前一半是相似图片对的第二张图片， 后一半是不相似图片对的第二张图片

idxa = repmat(1:size(sampleData,2), 1, 2);

%产生相似图片对的第二张图片的索引
idxb1 = zeros(1, size(idxa,2)/2);
t=0;
for i=1:length(SampleCountEachPerson)
    NumOfPerson = SampleCountEachPerson(i);
    
    idxb1(t+1: t+NumOfPerson) = randperm(NumOfPerson)+t;
    while(sum(idxb1(t+1: t+NumOfPerson) == idxa(t+1: t+NumOfPerson)) ~= 0)
        idxb1(t+1: t+NumOfPerson) = randperm(NumOfPerson)+t;
    end
    t = t+NumOfPerson;
end

%产生不相似图片对的第二张图片的索引
idxb2 = zeros(1, size(idxa,2)/2);
t=0;
SumOfSample = sum(SampleCountEachPerson);
for i=1:length(SampleCountEachPerson)
    NumOfPerson = SampleCountEachPerson(i);
    if(i==1)
        temp = NumOfPerson+1:SumOfSample;
    elseif(i==length(SampleCountEachPerson))
        temp = 1:SumOfSample-NumOfPerson;
    else
        p = sum(SampleCountEachPerson(1:i-1));
        temp = [1:p p+NumOfPerson+1:SumOfSample ];
    end
    temp = temp( randperm(length(temp)) );
    if(length(temp) < NumOfPerson)
        temp = repmat(temp, 1, NumOfPerson/length(temp)+1)
    end

    idxb2(t+1: t+NumOfPerson) = temp(1:NumOfPerson);
    t = t+NumOfPerson;
end

idxb = [idxb1 idxb2];

end
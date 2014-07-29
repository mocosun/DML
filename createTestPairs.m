function [idxa, idxb] = createPairs(sampleData, SampleCountEachPerson)
%idxa: 索引sampleData，前一半是相似图片对的第一张图片， 后一半是不相似图片对的第一张图片
%idxb: 索引sampleData，前一半是相似图片对的第二张图片， 后一半是不相似图片对的第二张图片

idxa = 1:size(sampleData,2);

%产生相似图片对的第二张图片的索引
idxb = zeros(1, size(idxa,2));
t=0;
for i=1:length(SampleCountEachPerson)
    NumOfPerson = SampleCountEachPerson(i);
    idxb(t+1: t+NumOfPerson) = randperm(NumOfPerson)+t;
    while(sum(idxb(t+1: t+NumOfPerson) == idxa(t+1: t+NumOfPerson)) ~= 0)
        idxb(t+1: t+NumOfPerson) = randperm(NumOfPerson)+t;
    end
    t = t+NumOfPerson;
end

 

end
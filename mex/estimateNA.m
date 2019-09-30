function data = estimateNaN(data,is_factor)

ncol = size(data,2);
class = unique(data(:,ncol));
nclass = length(class);
for i=1:ncol-1
    are_nans = isnan(data(:,i));
    if ~any(are_nans), continue; end
    
    for j=1:nclass
        this_class = data(:,ncol)==class(j);
        if ~any(this_class), continue; end
        
        valid = data(~are_nans & this_class,i);
        if isempty(valid), continue; end
        
        if any(are_nans & this_class)
            if is_factor(i)
                levels = unique(valid)'; % These are the levels that this class take
                [~,ii] = max(sum(bsxfun(@eq,valid,levels),1));
                data(are_nans & this_class,i) = levels(ii);
            else
                data(are_nans & this_class,i) = mean(valid);
            end
        end
    end
    % If after checking all the variables there are still nans, use a
    % global inputation method
    are_nans = isnan(data(:,i));
    if any(are_nans)
        valid = data(~are_nans,i);
        if is_factor(i)
            levels = unique(valid)'; % These are the levels that this class take
            [~,ii] = max(sum(bsxfun(@eq,valid,levels),1));
            data(are_nans,i) = levels(ii);
        else
            data(are_nans,i) = mean(valid);
        end
    end
end
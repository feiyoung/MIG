function  [Xmis,Xmiss, group, type, type_scale] = precheckdata(Xmis, group, type)

if(size(group,1)~=1)
    group = group';
end
ind_set = unique(group);
ng = length(ind_set);
if ~isempty(setdiff(1:ng, ind_set))
    error('ID number of types must match types!(type1: 1, type2: 2, etc.)')
end
if ng ~= size(type,1)
    error('The number of groups must match with length of variable type!');
end
if size(Xmis,2) ~= length(group)
    error('The columns of Xmis must match with length of group!');
end

% check zero-variance variables
stdVec = std(Xmis, 'omitnan');
var_id = find(stdVec < 1e-5);
if(~isempty(var_id))
    warning('on')
    warning('There are %d zero-variance variables in Xmis that will be removed as follows: ', length(var_id));
    warning(['They are ', num2str(var_id), ' -th variable']);
    Xmis = Xmis(:, stdVec>=1e-5);
    group = group(stdVec>=1e-5);
end
type_set = 1:ng;
for s = 1:ng
   if(sum(group==s)==0)
      type_set = setdiff(type_set, s);
    end 
end
type = type(type_set,:);
Xmiss = Xmis;

n = size(Xmis, 1);
type_scale = type;
for s = 1:ng % handle the problem that poisson variables or normal variable have large magnitude
    type_scale{s,2} = []; % it may induce instability of algorithm.
    switch type{s,1}
        case 'poisson'
          cutoff = 80;
          id_types = find(group==s);
          maxVec = max(Xmis(:, id_types));
          id = id_types(maxVec > cutoff);
          if(~isempty(id))
             Xmis(:, id) = Xmis(:,id) ./  repmat(maxVec(maxVec > cutoff), n, 1);
          end
          type_scale{s,2} =  [id; maxVec(maxVec > cutoff)];
        case 'normal' 
          cutoff = 10;
          id_types = find(group==s);
          stdVec = std(Xmis(:, id_types), 'omitnan');
          id = id_types(stdVec > cutoff);
          if(~isempty(id))
             Xmis(:, id) = Xmis(:,id) ./  repmat(stdVec(stdVec > cutoff), n, 1);
          end
          type_scale{s,2} =  [id; stdVec(stdVec > cutoff)];
    end
end

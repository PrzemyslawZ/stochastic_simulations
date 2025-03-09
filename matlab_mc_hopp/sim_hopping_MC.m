function nreg = sim_hopping_MC(N,hop_rate,loss_left,loss_right,gain_right,loss_chain,t_grid)
%This function uses MC techniques bosonic particles hopping stochastically and non-reciprocally
% on a chain, with injection and extraction on the edges, and losses along
% the chain.

%hop_rate: hopping rate
%loss_left: loss rate on the left end of the chain
%loss_right: loss rate on the right end of the chain
%gain_right: gain rate on the right end of the chain
%loss_cav: loss rate in the chain
%t_grid: list of time-steps


%%%%%%%%Initial parametersconfiguration and time grid%%%%%%
% This code does not simulate events at specific fixed times; rather, it
% computes the time elapsed between successive random events. For each new 
% event, we save both the configuration n and interval t in lists ns and
% ts; once we reached the final time, we use linear interpolation to
% regularize ns and ts according to the time grid t_grid.
    
    n=zeros(N,1); %initial chain configuration
    ns=[n,n];
    tend=t_grid(end); 
    t=0;    
    ts=[t];
    %fprintf('size A %s\n', mat2str(size(ns)));
    while (t<tend)
        %We list the probabilities of all possible events in a list A
        A(1:N-1)=hop_rate*n(2:N).*(1+n(1:N-1)); %Hopping events
        A(N)=loss_right*n(N); %losses on the right
        A(N+1)=gain_right*(1+n(N)); %injection on the right        
        A(N+2)=loss_left*n(1); %losses on the left
        A(N+3:2*N+2)=loss_chain*n(1:N);%losses along the chain
        %fprintf('size A %s\n', mat2str(A));
        %We draw the time elapsed until the next event
        r=rand(1);
        %fprintf('%f\n', sum(A));
        if sum(A)~=0
            t0=-log(r)/sum(A);      
            res=mnrnd(1,A/sum(A)); %generate random number from multinomial distribution that indicates which event takes place
        else 
            t0=tcutoff;
            fprintf(tcutoff);
            res=zeros(1,length(A));
        end
        t=t+t0;

        %Update the population
        dn=zeros(N,1);
        dn(1:N)=[res(1:N-1),0]-[0,res(1:N-1)]-res(N+3:2*N+2);
        dn(1)=dn(1)-res(N+2);
        dn(N)=dn(N)-res(N)+res(N+1);
        n=n+dn;        
        
        ns=[ns,n,n]; %Stores two copies of the population at each point in time
        ts=[ts,t-10^-8,t+10^-8]; %Stores the time vector for one traj
    end
    
fprintf('size t_grid %s\n', mat2str(size(t_grid)));
fprintf('size ts %s\n', mat2str(size(ts)));
ts=[ts,t+2*10^-8];
nreg=interp1(ts,ns.',t_grid); %population versus regularized time
function [xCurrent,Resnorm,FVAL,EXITFLAG,OUTPUT,LAMBDA,JACOB] = lsqcurvefit(FUN,xCurrent,XDATA,YDATA,LB,UB,A,B,Aeq,Beq,NONLCON,options,varargin)
%LSQCURVEFIT solves non-linear least squares problems.
%   LSQCURVEFIT attempts to solve problems of the form:
%   min  sum {(FUN(X,XDATA)-YDATA).^2}  where X, XDATA, YDATA and the values returned
%    X                                  by FUN can be vectors or matrices.
%   subject to:  A*X  <= B, Aeq*X  = Beq (linear constraints)
%                C(X) <= 0, Ceq(X) = 0  (nonlinear constraints)
%                LB <= X <= UB  (bounds)
%
%   LSQCURVEFIT implements three different algorithms: trust region reflective,
%   Levenberg-Marquardt, and Interior-Point. Choose one via the option Algorithm: for
%   instance, to  choose Levenberg-Marquardt, set
%   OPTIONS = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt'),
%   and then pass OPTIONS to LSQCURVEFIT.
%
%   X = LSQCURVEFIT(FUN,X0,XDATA,YDATA) starts at X0 and finds coefficients
%   X to best fit the nonlinear functions in FUN to the data YDATA (in the
%   least-squares sense). FUN accepts inputs X and XDATA and returns a
%   vector (or matrix) of function values F, where F is the same size as
%   YDATA, evaluated at X and XDATA. NOTE: FUN should return FUN(X,XDATA)
%   and not the sum-of-squares sum((FUN(X,XDATA)-YDATA).^2).
%   ((FUN(X,XDATA)-YDATA) is squared and summed implicitly in the
%   algorithm.)
%
%   X = LSQCURVEFIT(FUN,X0,XDATA,YDATA,LB,UB) defines a set of lower and
%   upper bounds on the design variables, X, so that the solution is in the
%   range LB <= X <= UB. Use empty matrices for LB and UB if no bounds
%   exist. Set LB(i) = -Inf if X(i) is unbounded below; set UB(i) = Inf if
%   X(i) is unbounded above.
%
%   X = LSQCURVEFIT(FUN,X0,XDATA,YDATA,LB,UB,A,B,Aeq,Beq) finds a minimum X to
%   the sum of squares of the functions in FUN subject to the linear inequalities
%   A*X <= B and linear equalities Aeq*X = Beq.
%
%   X = LSQCURVEFIT(FUN,X0,XDATA,YDATA,LB,UB,A,B,Aeq,Beq,NONLCON) subjects
%   the minimization to the constraints defined in NONLCON. The function
%   NONLCON accepts X and returns the vectors C and Ceq, representing the
%   nonlinear inequalities and equalities respectively. LSQCURVEFIT minimizes
%   FUN such that C(X) <= 0 and Ceq(X) = 0.
%
%   X = LSQCURVEFIT(FUN,X0,XDATA,YDATA,LB,UB,A,B,Aeq,Beq,NONLCON,OPTIONS) minimizes
%   with the default parameters replaced by values in OPTIONS, an argument created
%   with the OPTIMOPTIONS function. See OPTIMOPTIONS for details. Use the
%   SpecifyObjectiveGradient option to specify that FUN also returns a
%   second output argument J that is the Jacobian matrix at the point X. If
%   FUN returns a vector F of m components when X has length n, then J is
%   an m-by-n matrix where J(i,j) is the partial derivative of F(i) with
%   respect to x(j). (Note that the Jacobian J is the transpose of the
%   gradient of F.)
%
%   X = LSQCURVEFIT(PROBLEM) solves the non-linear least squares problem
%   defined in PROBLEM. PROBLEM is a structure with the function FUN in
%   PROBLEM.objective, the start point in PROBLEM.x0, the 'xdata' in
%   PROBLEM.xdata, the 'ydata' in PROBLEM.ydata, the linear inequality
%   constraints in PROBLEM.Aineq and PROBLEM.bineq, the linear equality constraints
%   in PROBLEM.Aeq and PROBLEM.beq, the lower bounds in PROBLEM.lb, the upper bounds
%   in PROBLEM.ub, the nonlinear constraint function in PROBLEM.nonlcon, the options
%   structure in PROBLEM.options, and solver name 'lsqcurvefit' in PROBLEM.solver.
%
%   [X,RESNORM] = LSQCURVEFIT(FUN,X0,XDATA,YDATA,...) returns the value of
%   the squared 2-norm of the residual at X: sum {(FUN(X,XDATA)-YDATA).^2}.
%
%   [X,RESNORM,RESIDUAL] = LSQCURVEFIT(FUN,X0,...) returns the value of
%   residual, FUN(X,XDATA)-YDATA, at the solution X.
%
%   [X,RESNORM,RESIDUAL,EXITFLAG] = LSQCURVEFIT(FUN,X0,XDATA,YDATA,...)
%   returns an EXITFLAG that describes the exit condition. Possible values
%   of EXITFLAG and the corresponding exit conditions are listed below. See
%   the documentation for a complete description.
%
%     1  LSQCURVEFIT converged to a solution.
%     2  Change in X too small.
%     3  Change in RESNORM too small.
%     4  Computed search direction too small.
%     0  Too many function evaluations or iterations.
%    -1  Stopped by output/plot function.
%    -2  Bounds are inconsistent.
%
%   [X,RESNORM,RESIDUAL,EXITFLAG,OUTPUT] = LSQCURVEFIT(FUN,X0,XDATA,
%   YDATA,...) returns a structure OUTPUT with the number of iterations
%   taken in OUTPUT.iterations, the number of function evaluations in
%   OUTPUT.funcCount, the algorithm used in OUTPUT.algorithm, the number of
%   CG iterations (if used) in OUTPUT.cgiterations, the first-order
%   optimality (if used) in OUTPUT.firstorderopt, and the exit message in
%   OUTPUT.message.
%
%   [X,RESNORM,RESIDUAL,EXITFLAG,OUTPUT,LAMBDA] = LSQCURVEFIT(FUN,X0,XDATA,
%   YDATA,...) returns the set of Lagrangian multipliers, LAMBDA, at the
%   solution: LAMBDA.lower for LB and LAMBDA.upper for UB.
%
%   [X,RESNORM,RESIDUAL,EXITFLAG,OUTPUT,LAMBDA,JACOBIAN] = LSQCURVEFIT(FUN,
%   X0,XDATA,YDATA,...) returns the Jacobian of FUN at X.
%
%   Examples
%     FUN can be specified using @:
%        xdata = [5;4;6];          % example xdata
%        ydata = 3*sin([5;4;6])+6; % example ydata
%        x = lsqcurvefit(@myfun, [2 7], xdata, ydata)
%
%   where myfun is a MATLAB function such as:
%
%       function F = myfun(x,xdata)
%       F = x(1)*sin(xdata)+x(2);
%
%   FUN can also be an anonymous function:
%       x = lsqcurvefit(@(x,xdata) x(1)*sin(xdata)+x(2),[2 7],xdata,ydata)
%
%   If FUN is parameterized, you can use anonymous functions to capture the
%   problem-dependent parameters. Suppose you want to solve the
%   curve-fitting problem given in the function myfun, which is
%   parameterized by its second argument c. Here myfun is a MATLAB file
%   function such as
%
%       function F = myfun(x,xdata,c)
%       F = x(1)*exp(c*xdata)+x(2);
%
%   To solve the curve-fitting problem for a specific value of c, first
%   assign the value to c. Then create a two-argument anonymous function
%   that captures that value of c and calls myfun with three arguments.
%   Finally, pass this anonymous function to LSQCURVEFIT:
%
%       xdata = [3; 1; 4];           % example xdata
%       ydata = 6*exp(-1.5*xdata)+3; % example ydata
%       c = -1.5;                    % define parameter
%       x = lsqcurvefit(@(x,xdata) myfun(x,xdata,c),[5;1],xdata,ydata)
%
%   See also OPTIMOPTIONS, LSQNONLIN, FSOLVE, @.

%   Copyright 1990-2022 The MathWorks, Inc.

arguments
    FUN = []
    xCurrent = []
    XDATA = []
    YDATA = []
    LB = []
    UB = []
    A = []
    B = []
    Aeq = []
    Beq = []
    NONLCON = []
    options = []
end
arguments (Repeating)
    varargin
end

% ------------Initialization----------------
defaultopt = struct(...
    'Algorithm','trust-region-reflective',...
    'DerivativeCheck','off',...
    'Diagnostics','off',...
    'DiffMaxChange',Inf,...
    'DiffMinChange',0,...
    'Display','final',...
    'FinDiffRelStep', [], ...
    'FinDiffType','forward',...
    'FunValCheck','off',...
    'GradConstr','off',...
    'InitDamping', 0.01, ...
    'Jacobian','off',...
    'JacobMult',[],...
    'JacobPattern','sparse(ones(Jrows,Jcols))',...
    'MaxFunEvals',[],...
    'MaxIter',400,...
    'MaxPCGIter','max(1,floor(numberOfVariables/2))',...
    'OutputFcn',[],...
    'PlotFcns',[],...
    'PrecondBandWidth',Inf,...
    'ProblemdefOptions', struct, ...    
    'ScaleProblem','none',...
    'TolCon', 1e-6,...
    'TolFun', 1e-6,...
    'TolFunValue', 1e-6, ...
    'TolPCG',0.1,...
    'TolX',1e-6,...
    'TypicalX','ones(numberOfVariables,1)',...
    'UseParallel',false, ...
    'BarrierParamUpdate', 'monotone', ...
    'InitBarrierParam', 0.1, ...
    'SubproblemAlgorithm', 'ldl-factorization');

% If just 'defaults' passed in, return the default options in X
if nargin==1 && nargout <= 1 && strcmpi(FUN,'defaults')
    xCurrent = defaultopt;
    return
end

problemInput = false;
if nargin == 1
    if isa(FUN,'struct')
        problemInput = true;
        [FUN,xCurrent,XDATA,YDATA,LB,UB,A,B,Aeq,Beq,NONLCON,options] = separateOptimStruct(FUN);
    else % Single input and non-structure.
        error(message('optimlib:lsqcurvefit:InputArg'));
    end
end

% Set the default ProblemdefOptions
defaultopt.ProblemdefOptions = ...
    optim.internal.constants.DefaultOptionValues.ProblemdefOptions;

if nargin < 4 && ~problemInput
    error(message('optimlib:lsqcurvefit:NotEnoughInputs'));
end

caller = 'lsqcurvefit';

% 2 <= numInputs <= 5 ==> oldAPI
% numInputs = 5 is the middle case (any api)
if ~problemInput
    numInputs = nargin - 2; % exclude xdata, ydata; the lsq algo check is common to lsqnonlin
else
    numInputs = 1; % exclude xdata, ydata
end
[boundsOnlySolvers,optimgetFlag,LB,UB,A,B,Aeq,Beq,NONLCON,options,varargin] ...
    = lsqAlgorithmCheck(caller,LB,UB,A,B,Aeq,Beq,NONLCON,options,defaultopt,numInputs,varargin{:});

% Check for non-double inputs
msg = isoptimargdbl('LSQCURVEFIT', {'X0','YDATA','LB','UB','A','B','Aeq','Beq'}, ...
                                    xCurrent,YDATA,LB,UB,A,B,Aeq,Beq);
if ~isempty(msg)
    error('optimlib:lsqcurvefit:NonDoubleInput',msg);
end

% Prepare the options for the solver
options = prepareOptionsForSolver(options, caller);
    
% Use the constrained solver
if ~boundsOnlySolvers
    [xCurrent,Resnorm,FVAL,EXITFLAG,OUTPUT,LAMBDA,JACOB] = cnls(caller, @(X) feval(FUN, X, XDATA) - YDATA,...
        xCurrent,A,B,Aeq,Beq,LB,UB,NONLCON,options,defaultopt,optimgetFlag);
    return
end

[funfcn_x_xdata,mtxmpy_xdata,flags,sizes,~,xstart,lb,ub,EXITFLAG, ...
    Resnorm,FVAL,LAMBDA,JACOB,OUTPUT,earlyTermination] = ...
    lsqnsetup(FUN,xCurrent,LB,UB,options,defaultopt,optimgetFlag, ...
    caller,nargout,length(varargin));
if earlyTermination
    return % premature return because of problem detected in lsqnsetup()
end

xCurrent(:) = xstart; % reshape back to user shape before evaluation
funfcn = funfcn_x_xdata; % initialize user functions funfcn, which depend only on x
% Catch any error in user objective during initial evaluation only
switch funfcn_x_xdata{1}
    case 'fun'
        try
            initVals.F = feval(funfcn_x_xdata{3},xCurrent,XDATA,varargin{:});
        catch userFcn_ME
            userFcn_ME = initEvalErrorHandler(userFcn_ME,funfcn_x_xdata{3}, ...
                'optimlib:lsqcurvefit:InvalidFUN');
            rethrow(userFcn_ME);
        end
        initVals.J = [];
        funfcn{3} = @objective;
    case 'fungrad'
        try
            [initVals.F,initVals.J] = feval(funfcn_x_xdata{3},xCurrent,XDATA,varargin{:});
        catch userFcn_ME
            userFcn_ME = initEvalErrorHandler(userFcn_ME,funfcn_x_xdata{3}, ...
                'optimlib:lsqcurvefit:InvalidFUN');
            rethrow(userFcn_ME);
        end
        funfcn{3} = @objectiveAndJacobian;
        funfcn{4} = @objectiveAndJacobian;
    case 'fun_then_grad'
        try
            initVals.F = feval(funfcn_x_xdata{3},xCurrent,XDATA,varargin{:});
        catch userFcn_ME
            userFcn_ME = initEvalErrorHandler(userFcn_ME,funfcn_x_xdata{3}, ...
                'optimlib:lsqcurvefit:InvalidFUN');
            rethrow(userFcn_ME);
        end
        try
            initVals.J = feval(funfcn_x_xdata{4},xCurrent,XDATA,varargin{:});
        catch userFcn_ME
            userFcn_ME = initEvalErrorHandler(userFcn_ME,funfcn_x_xdata{4}, ...
                'optimlib:lsqcurvefit:InvalidFUNJac');
            rethrow(userFcn_ME);
        end
        funfcn{3} = @objective;
        funfcn{4} = @jacobian;
    otherwise
        error(message('optimlib:lsqcurvefit:UndefCallType'))
end

if ~isequal(size(initVals.F),size(YDATA))
    error(message('optimlib:lsqcurvefit:YdataSizeMismatchFunVal'))
end
initVals.F = initVals.F - YDATA; % preserve initVals.F shape until after subtracting YDATA

% Flag to determine whether to look up the exit msg.
flags.makeExitMsg = logical(flags.verbosity) || nargout > 4;

% Pass functions that depend only on x: funfcn and jacobmult
[xCurrent,Resnorm,FVAL,EXITFLAG,OUTPUT,LAMBDA,JACOB] = ...
    lsqncommon(funfcn,xCurrent,lb,ub,options,defaultopt,optimgetFlag,caller,...
    initVals,sizes,flags,@jacobmult,varargin{:});

% Add fields that are relevant for the constrained solvers.
OUTPUT.bestfeasible = [];
OUTPUT.constrviolation = [];
LAMBDA.eqlin = [];
LAMBDA.ineqlin = [];
LAMBDA.eqnonlin = [];
LAMBDA.ineqnonlin = [];

%-------------------------------------------------------------------------------
% Nested functions that depend only on x and capture the constant values
% xdata and ydata, and also varargin
    function F = objective(x,varargin)
        F = feval(funfcn_x_xdata{3},x,XDATA,varargin{:});
        F = F - YDATA;
    end
    function [F,J] = objectiveAndJacobian(x,varargin)
        % Function value and Jacobian returned together
        [F,J] = feval(funfcn_x_xdata{3},x,XDATA,varargin{:});
        F = F - YDATA;
    end
    function J = jacobian(x,varargin)
        % Jacobian returned in separate function
        J = feval(funfcn_x_xdata{4},x,XDATA,varargin{:});
    end
    function W = jacobmult(Jinfo,Y,flag,varargin)
        W = feval(mtxmpy_xdata,Jinfo,Y,flag,XDATA,varargin{:});
    end
end
%-------------------------------------------------------------------------------
function exception = initEvalErrorHandler(exception,fcnIn,errID)

if (isa(fcnIn,'function_handle'))
    nInputs = nargin(fcnIn);
    if nInputs > 0 && nInputs ~= 2
        error(message('optimlib:lsqcurvefit:NotEnoughInputArg'));
    end
end
optim_ME = MException(message(errID));
exception = addCause(exception,optim_ME);

end
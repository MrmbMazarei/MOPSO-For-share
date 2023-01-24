# MOPSO-For-share
Multi Objective PSO
!!This program is written for input variable selection by Multy objective PSO optimization algorithm
!!in this prog Neural network is written by PSO algorithem.
program Mo_PSO_IVS
implicit none

!!Variable introduction==============================================================================================================
real,dimension(12,41)::Input1,Input2,Input3,Input4,Input5,Input6,Input7,Input8,Input9,Input10,Input11,Input12,Output1

real::a1,b1,a2,b2,sum1,sum2,Ytr_mean,Yts_mean,MaxY,MinY,rand1,rand2,Rand_IW,Rand_LW,Rand_b,NeronOutput2,MSEj,NSEj,Cpj,Cgj,Wj,MSE_p,&
NSE_p,Cp_p,Cg_p,W_p,MSE_g,NSE_g,Cp_g,Cg_g,W_g,w_p_star,cp_p_star,cg_p_star,w_g_star,cp_g_star,cg_g_star,fi_P,fi_g,SUM3,SUM4,R2,&
Ytr_P_mean,SUM5,SUM6,Yts_P_mean,Yval_P_Mean,a3,b3,Mutation_Rate,rand3,sum7,sum8,sum9,Yval_mean

real,ALLOCATABLE::Xin(:,:),Yout(:),Xtr(:,:),Ytr(:),Xts(:,:),Yts(:),XN(:,:),YN(:),MaxX(:),MinX(:),SelectionMatrix(:,:),IW(:,:),&
LW(:,:),b(:,:),W(:),Cp(:),Cg(:),NeronOutput1(:),Ytr_Pre(:),Yts_Pre(:),Error(:),MSE(:),NSE(:),IWj(:),LWj(:),bj(:),IW_p(:),LW_p(:),&
b_p(:),IW_g(:),LW_g(:),b_g(:),V_IW(:,:),V_LW(:,:),V_b(:,:),SelectionMatrix_Output_valid(:,:),SelectionMatrix_Output_train(:,:),&
Total_Fronts(:,:,:),Total_Frontsj(:),Input_Cunt(:),crowding_dis(:,:),SelectionMatrix_p(:),SelectionMatrix_Output_train_p(:),&
SelectionMatrix_Output_valid_p(:),SelectionMatrix_g(:),SelectionMatrix_Output_train_g(:),SelectionMatrix_Output_valid_g(:),&
W_selection(:),Cp_selection(:),Cg_selection(:),v_selectionmatrix(:,:),sv(:,:),Xval(:,:),Yval(:),Yval_pre(:),SelectionMatrix_Output_&
test(:,:),SelectionMatrix_Output_test_p(:),SelectionMatrix_Output_test_g(:)


integer::i,j,k,l,m,n,o,SWARM_VariableSelection,SWARM_NN_train,SereisNum,DataNum,InputNum,OutputNum,TrainPerc,TrainNumber,TestNumber,&
         repet_selection,repet_NN_train,ii,jj,kk,ll,NumOfObjFunc,mu_no,mu_selected,ValidPerc,ValidNumber

integer,ALLOCATABLE::Sp(:,:),Nq(:),Front(:)

!!Parameters==========================================================================================================================
CALL RANDOM_SEED()

SWARM_VariableSelection=50 !Optimal Swarm is (100) , this is the swarm for input variable algorithem
SWARM_NN_train=50 !Optimal Swarm is (100) , this is the swarm for nural network training algorithem

repet_selection=500 !iteration number for the best variables selection
repet_NN_train=1000 !iteration for NN training

DataNum=12*41 !Month*Years
InputNum=12 !Input Series Number
OutputNum=1 !Output Series Number
SereisNum=InputNum+OutputNum !total input and output series number
NumOfObjFunc=2 !Number of objective function for input selection

TrainPerc=60 !percent of data that you want to use for train
TrainNumber=INT(TrainPerc*12*41/100)
ValidPerc=20
ValidNumber=INT(ValidPerc*12*41/100)
TestNumber=12*41-TrainNumber-ValidNumber

a1=-1 !Min Number For Weight
b1=1 !Max Number For Weight
a2=-5 !Min Number For Bias
b2=5 !Max Number For Bias
a3=0
b3=1

Mutation_Rate=0.1
!!Allocation====================================================================================================================
Allocate(Xin(InputNum,DataNum),Yout(DataNum),XN(InputNum,12*41),YN(12*41),Xts(InputNum,TestNumber),Yts(TestNumber),&
Xtr(InputNum,trainNumber),Ytr(trainNumber),MaxX(InputNum),MinX(InputNum),SelectionMatrix(SWARM_VariableSelection,InputNum),&
IW(SWARM_NN_train,(InputNum**2)),LW(SWARM_NN_train,InputNum),b(SWARM_NN_train,(1+InputNum)),W(SWARM_NN_train),Cp(SWARM_NN_train),&
Cg(SWARM_NN_train),NeronOutput1(InputNum),Ytr_Pre(trainNumber),Yts_Pre(TestNumber),Input_Cunt(SWARM_VariableSelection),&
Error(SWARM_NN_train),MSE(SWARM_NN_train),NSE(SWARM_NN_train),IWj((InputNum**2)),LWj(InputNum),bj(1+InputNum),IW_p(InputNum**2),&
LW_p(InputNum),b_p(InputNum+1),IW_g(InputNum**2),LW_g(InputNum),b_g(InputNum+1),V_IW(SWARM_NN_train,InputNum**2),V_LW&
(SWARM_NN_train,InputNum),V_b(SWARM_NN_train,InputNum+1),SelectionMatrix_Output_valid(SWARM_VariableSelection,NumOfObjFunc),&
SelectionMatrix_Output_train(SWARM_VariableSelection,NumOfObjFunc),Sp(SWARM_VariableSelection,SWARM_VariableSelection-1),Nq&
(SWARM_VariableSelection),Front(SWARM_VariableSelection),Total_Fronts(SWARM_VariableSelection,InputNum+2*NumOfObjFunc,SWARM_&
VariableSelection+1),Total_Frontsj(InputNum+2*NumOfObjFunc),crowding_dis(SWARM_VariableSelection,SWARM_VariableSelection),&
SelectionMatrix_p(InputNum),SelectionMatrix_Output_train_p(NumOfObjFunc),SelectionMatrix_Output_valid_p(NumOfObjFunc),&
SelectionMatrix_g(InputNum),SelectionMatrix_Output_train_g(NumOfObjFunc),SelectionMatrix_Output_valid_g(NumOfObjFunc),W_selection&
(SWARM_VariableSelection),Cp_selection(SWARM_VariableSelection),Cg_selection(SWARM_VariableSelection),v_selectionmatrix(&
SWARM_VariableSelection,InputNum),sv(SWARM_VariableSelection,InputNum),Xval(InputNum,ValidNumber),Yval(ValidNumber),Yval_pre(Valid&
Number),SelectionMatrix_Output_test(SWARM_VariableSelection,NumOfObjFunc),SelectionMatrix_Output_test_p(NumOfObjFunc),&
SelectionMatrix_Output_test_g(NumOfObjFunc))

!!Data Loading==================================================================================================================
open(1,file='R_Sarab.txt',status='old') 
read(1,*)input1
open(2,file='R_SaeadAbad.txt',status='old') 
read(2,*)input2
open(3,file='R_Vaniar.txt',status='old') 
read(3,*)input3
open(4,file='R_Esbaghran.txt',status='old') 
read(4,*)input4
open(5,file='R_Nahand.txt',status='old') 
read(5,*)input5
open(7,file='R_Sohzab.txt',status='old') 
read(7,*)input6
open(8,file='R_Ghourigol.txt',status='old') 
read(8,*)input7
open(9,file='R_Ghoshchi.txt',status='old') 
read(9,*)input8
open(10,file='R_BostanAbad.txt',status='old') 
read(10,*)input9
open(11,file='T_Srab.txt',status='old') 
read(11,*)input10
open(12,file='T_Mirkoh.txt',status='old') 
read(12,*)input11
open(13,file='T_Ghourigol.txt',status='old') 
read(13,*)input12
open(14,file='D_Vaniar.txt',status='old') 
read(14,*)output1
open(100,file='Resault.txt',status='old')
!Input Data organization===============================================================================================
do i=1,SereisNum
   do j=1,41
      do k=1,12
	     if (i==1) then
		    Xin(i,(j-1)*12+k)=input1(k,j)
		 else if (i==2) then
		    Xin(i,(j-1)*12+k)=input2(k,j)
		 else if (i==3) then
		    Xin(i,(j-1)*12+k)=input3(k,j)
		 else if (i==4) then
		    Xin(i,(j-1)*12+k)=input4(k,j)
		 else if (i==5) then
		    Xin(i,(j-1)*12+k)=input5(k,j)
		 else if (i==6) then
		    Xin(i,(j-1)*12+k)=input6(k,j)
		 else if (i==7) then
		    Xin(i,(j-1)*12+k)=input7(k,j)
		 else if (i==8) then
		    Xin(i,(j-1)*12+k)=input8(k,j)
		 else if (i==9) then
		    Xin(i,(j-1)*12+k)=input9(k,j)
		 else if (i==10) then
		    Xin(i,(j-1)*12+k)=input10(k,j)
		 else if (i==11) then
		    Xin(i,(j-1)*12+k)=input11(k,j)
		 else if (i==12) then
		    Xin(i,(j-1)*12+k)=input12(k,j)
		 else if (i==13) then
		    Yout((j-1)*12+k)=output1(k,j)
		 end if
	  end do
   end do
end do

!!Normalization==================================================================================================================
!Normalization between 0~1
Do i=1,InputNum
   MaxX(i)=Xin(i,1)  !Maximum matrix for either input time series
   Do j=2,DataNum
	  if (MaxX(i)<Xin(i,j)) then 
		  MaxX(i)=Xin(i,j)
	  End if
   End do
End Do

   MaxY=Yout(1)  !Maximum matrix for either output time series
   Do j=2,DataNum
	  if (MaxY<Yout(j)) then 
		  MaxY=Yout(j)
	  End if
   End do


Do i=1,InputNum
   MinX(i)=Xin(i,1)  !Minimum matrix for either input time series
   Do j=2,DataNum
	  if (MinX(i)>Xin(i,j)) then 
		  MinX(i)=Xin(i,j)
	  End if
   End do
End Do

   MinY=Yout(1)  !Minimum matrix for either output time series
   Do j=2,DataNum
	  if (MinY>Yout(j)) then 
		  MinY=Yout(j)
	  End if
   End do
		 
Do i=1,InputNum
   Do j=1,DataNum
	  XN(i,j)=(Xin(i,j)-MinX(i))/(MaxX(i)-MinX(i)) !Normalized input matrix
   End do
End do

   Do j=1,DataNum
	  YN(j)=(Yout(j)-MinY)/(MaxY-MinY) !Normalized output matrix
   End do

!!in this section we separate training data from test data and validation data=============================================================
!!Train Data:
Xtr(:,:)=XN(:,1:TrainNumber) !training output data
Ytr(:)=YN(1:TrainNumber) !training input data

!!in this section calculate statistical parameters
Do i=1,TrainNumber
   Ytr_mean=Ytr_mean+Ytr(i)
End do
Ytr_mean=Ytr_mean/TrainNumber

Do i=1,TrainNumber
   sum1=sum1+((Ytr(i)-Ytr_mean)**2)
End do

!!Validation Data------------------------------------------------------------------------------
Xval(:,:)=XN(:,TrainNumber+1:TrainNumber+ValidNumber) !Test input data
Yval(:)=YN(TrainNumber+1:TrainNumber+ValidNumber) !Tast output data

!!in this section calculate statistical parameters
Do i=1,ValidNumber
   Yval_mean=Yval_mean+Yval(i)
End do
Yval_mean=Yval_mean/ValidNumber

Do i=1,ValidNumber
   sum7=sum7+((Yval(i)-Yval_mean)**2)
End do

!!Test Data------------------------------------------------------------------------------
Xts(:,:)=XN(:,TrainNumber+ValidNumber+1:492) !Test input data
Yts(:)=YN(TrainNumber+ValidNumber+1:492) !Tast output data

!!in this section calculate statistical parameters
Do i=1,TestNumber
   Yts_mean=Yts_mean+Yts(i)
End do
Yts_mean=Yts_mean/TestNumber

Do i=1,TestNumber
   sum2=sum2+((Yts(i)-Yts_mean)**2)
End do

!!the start of PSO input variable selection:===========================================================================
!First random selected variable
Do i=1,SWARM_VariableSelection
   Input_Cunt(i)=0
   Do j=1,InputNum
      Call random_number(rand1)
      if (rand1>0.5) then
         SelectionMatrix(i,j)=1
	     Input_Cunt(i)=Input_Cunt(i)+1
      else
         SelectionMatrix(i,j)=0
      end if
   End do !j

!parameters random selection: -------------------------------------------------------------------
   Call random_number(rand2)
        W_selection(i)=rand2 !inertia weights 
   Call random_number(rand2)
        Cp_selection(i)=rand2 !the inertia weight for local optimum
   Call random_number(rand2)
        Cg_selection(i)=rand2 !the inertia weight for global optimum

End do !i



!!simulation the Swarm===========================================================================================
DO i=1,repet_selection
   !write(*,*)SelectionMatrix(i,:),Input_Cunt(i)
   do j=1,SWARM_VariableSelection
  
!!RandomWeights===========================================================================
!Random 1st Swarm Weights for neural network

   IW(:,:)=0
   LW(:,:)=0
   b(:,:)=0
   
   DO k=1,SWARM_NN_train
      DO l=1,((Input_Cunt(j))**2)
	     Call Random_number(Rand_IW)
         IW(k,l)=a1+(b1-a1)*Rand_IW !the weights of inner layers
      ENd do !l
	  
      DO l=1,Input_Cunt(j)
         Call Random_number(Rand_LW)
         LW(k,l)=a1+(b1-a1)*Rand_LW !the weights of output layers
      End do !l
	  
      DO l=1,(1+Input_Cunt(j))
         Call Random_number(Rand_b)
         b(k,l)=a2+(b2-a2)*Rand_b !the curency of biases
      End do !l

	  !parameters random selection: -------------------------------------------------------------------
	  Call random_number(rand2)
           W(k)=rand2 !inertia weights 
      Call random_number(rand2)
           Cp(k)=rand2 !the inertia weight for local optimum
      Call random_number(rand2)
           Cg(k)=rand2 !the inertia weight for global optimum
		   !write(*,*)k,W(k),Cp(k),Cg(k)

   END DO !k
   

!!Training & PSO Optimization====================================================================
   DO k=1,repet_NN_train !the number of iteration for training network
!================================================================================================
      DO l=1,SWARM_NN_train !Number of swarm for training NN
         DO m=1,trainnumber
            CALL Structure_output !the structure of nural network
	        Ytr_Pre(m)=NeronOutput2 !output of nural network
			!write(*,*)m,Ytr(m),Ytr_Pre(m)
         End do !m
         CALL ERROR_Calculator
		 !write(*,*)MSE(l),NSE(l)
      End do !l
	  CALL SORTing
	  !write(*,*)MSE(1),NSE(1)

	  IW_P(:)=IW(1,:)
      LW_P(:)=LW(1,:)
      b_P(:)=b(1,:)
      MSE_P=MSE(1)
      NSE_P=NSE(1)
      W_P_Star=W(1)
      Cp_P_Star=Cp(1)
      Cg_P_Star=Cg(1)

	  if (k==1) then
         IW_g(:)=IW(1,:)
         LW_g(:)=LW(1,:)
         b_g(:)=b(1,:)
         MSE_g=MSE(1)
         NSE_g=NSE(1)
         W_g_Star=W(1)
         Cp_g_Star=Cp(1)
         Cg_g_Star=Cg(1)
      else if (MSE_P<MSE_g) then
              IW_g(:)=IW_P(:)
              LW_g(:)=LW_P(:)
              b_g(:)=b_P(:)
              MSE_g=MSE_P
              NSE_g=NSE_P
	          W_g_Star=W_P_Star
              Cp_g_Star=Cp_P_Star
              Cg_g_Star=Cg_P_Star
      end if
	  !write(*,*)MSE_P,MSE_g,NSE_P,NSE_g

!Velocity Update:  --------------------------------------------------------------------
!Update Inner layer weights
      do l=1,SWARM_NN_train
         do m=1,((Input_Cunt(j))**2)
            CALL random_number(fi_P)
	        CALL random_number(fi_g)
            V_IW(l,m)=W(l)*V_IW(l,m)+fi_P*Cp(l)*(IW_P(l)-IW(l,m))+fi_g*Cg(l)*(IW_g(m)-IW(l,m)) !the formula of velocity opdate
	        if (abs(b1-a1)<V_IW(l,m)) then
	           V_IW(l,m)=abs(b1-a1) !the limits of velocity
	        elseif (-abs(b1-a1)>V_IW(l,m)) then
	           V_IW(l,m)=-abs(b1-a1) !the limits of velocity
	        end if
	           IW(l,m)=IW(l,m)+V_IW(l,m) !weights place update
         End do !m
!Update output layer weights
		 do m=1,Input_Cunt(j)
            CALL random_number(fi_P)
	        CALL random_number(fi_g)
            V_LW(l,m)=W(l)*V_LW(l,m)+fi_P*Cp(l)*(LW_P(m)-LW(l,m))+fi_g*Cg(l)*(LW_g(m)-LW(l,m))
	        if (abs(b1-a1)<V_LW(l,m)) then
	           V_LW(l,m)=abs(b1-a1)
	        elseif (-abs(b1-a1)>V_LW(l,m)) then
	           V_LW(l,m)=-abs(b1-a1)
	        end if
	           LW(l,m)=LW(l,m)+V_LW(l,m)
         End do !m
!update Biases weights
         do m=1,Input_Cunt(j)+1
            CALL random_number(fi_P)
	        CALL random_number(fi_g)
            V_b(l,m)=W(l)*V_b(l,m)+fi_P*Cp(l)*(b_P(m)-b(l,m))+fi_g*Cg(l)*(b_g(m)-b(l,m))
	        if (abs(b2-a2)<V_b(l,m)) then
	           V_b(l,m)=abs(b2-a2)
	        elseif (-abs(b2-a2)>V_b(l,m)) then
	           V_b(l,m)=-abs(b2-a2)
	        end if
	        b(l,m)=b(l,m)+V_b(l,m)
         End do
      End do !l
   end do !k
   !Write(*,*)'MSE_Globally',MSE_g,NSE_g !MSE ERROR Globally

!Training output:--------------------------------------------------------------------------
   IW(1,:)=IW_g(:)
   LW(1,:)=LW_g(:)
   b(1,:)=b_g(:)
   l=1
   SUM3=0
   SUM4=0
   Ytr_P_Mean=0
   DO m=1,trainnumber
      CALL Structure_output
      Ytr_Pre(m)=NeronOutput2
      Ytr_P_Mean=Ytr_P_Mean+Ytr_Pre(m) !mean of neural network output for training set
      !write(*,*)m,Ytr_Pre(m),Ytr(m)
   End do !m
   Ytr_P_Mean=Ytr_P_Mean/trainnumber !mean of neural network output for training set

   DO m=1,trainnumber
      SUM3=SUM3+(Ytr_Pre(m)-Ytr_P_Mean)**2
      SUM4=SUM4+(Ytr(m)-Ytr_Mean)*(Ytr_Pre(m)-Ytr_P_Mean)
   End do
   R2=(sum4**2)/sum3/sum1 !R2 calculation

   CALL ERROR_Calculator
   !write(*,*)'Train Error',R2,NSE(l),MSE(l)

   SelectionMatrix_Output_train(j,1)=NSE(l)
   SelectionMatrix_Output_train(j,2)=MSE(l)

!Validation output:--------------------------------------------------------------------------
   l=1
   sum8=0
   sum9=0
   Yval_P_Mean=0
   DO m=1,ValidNumber
      CALL Valid_output
      Yval_Pre(m)=NeronOutput2
      Yval_P_Mean=Yval_P_Mean+Yval_Pre(m)
      !write(*,*)Yval_Pre(m),Yval(m)
   End do

   Yval_P_Mean=Yval_P_Mean/ValidNumber
   DO m=1,ValidNumber
      SUM8=SUM8+(Yval_Pre(m)-Yval_P_Mean)**2
      SUM9=SUM9+(Yval(m)-Yval_Mean)*(Yval_Pre(m)-Yval_P_Mean)
   End do
   R2=(sum9**2)/sum8/sum7

   CALL Valid_ERROR_Calculator
   !write(*,*)'Valid Error',R2,NSE(l),MSE(l)

   SelectionMatrix_Output_Valid(j,1)=NSE(l)
   SelectionMatrix_Output_Valid(j,2)=MSE(l)

!Test output:--------------------------------------------------------------------------
   l=1
   sum5=0
   sum6=0
   Yts_P_Mean=0
   DO m=1,testNumber
      CALL Test_output
      Yts_Pre(m)=NeronOutput2
      Yts_P_Mean=Yts_P_Mean+Yts_Pre(m)
      !write(*,*)Yts_Pre(m),Yts(m)
   End do

   Yts_P_Mean=Yts_P_Mean/TestNumber
   DO m=1,TestNumber
      SUM5=SUM5+(Yts_Pre(m)-Yts_P_Mean)**2
      SUM6=SUM6+(Yts(m)-Yts_Mean)*(Yts_Pre(m)-Yts_P_Mean)
   End do
   R2=(sum6**2)/sum5/sum2

   CALL Test_ERROR_Calculator
   !write(*,*)'Test Error',R2,NSE(k),MSE(k)

   SelectionMatrix_Output_test(j,1)=NSE(l)
   SelectionMatrix_Output_test(j,2)=MSE(l)

   End do !j

   Call Paretto_SORTing
   !write(*,*)Nq(:)
!Determine Fronts=====================================================================================================
   Do j=1,SWARM_VariableSelection !this counter is for front number
      l=0
      Do k=1,SWARM_VariableSelection
	     if ((Nq(k)-j+1)==0) then
		    Front(k)=j
			l=l+1
			Total_Fronts(l,1:InputNum,j)=SelectionMatrix(k,:)
			Total_Fronts(l,InputNum+1:InputNum+NumOfObjFunc,j)=SelectionMatrix_Output_train(k,:)
			Total_Fronts(l,InputNum+NumOfObjFunc+1:InputNum+2*NumOfObjFunc,j)=SelectionMatrix_Output_Valid(k,:)
		 end if
	  End do !k
	  !write(*,*)Total_Fronts(:,:,j+1)
   End do !j

   DO j=1,SWARM_VariableSelection+1
      DO k=1,SWARM_VariableSelection
         DO l=k+1,SWARM_VariableSelection
            if (Total_Fronts(l,InputNum+1,j)>Total_Fronts(k,InputNum+1,k)) then
			   if (Total_Fronts(l,InputNum+1,j).ne.0) then

	              Total_Frontsj(:)=Total_Fronts(k,:,j)
		          Total_Fronts(k,:,j)=Total_Fronts(l,:,j)
		          Total_Fronts(l,:,j)=Total_Frontsj(:)

			   END if
	        END if
         END DO
		 !write(*,*)j,k
		 !write(*,*)Total_Fronts(k,:,j)
      END DO
   END DO  
   
!crowding distance calculation========================================================================================
do j=1,SWARM_VariableSelection
   do k=1,SWARM_VariableSelection
      if ((k==1).and.(Total_Fronts(k,InputNum+1,j).ne.0)) then
	     crowding_dis(k,j)=20000
	  else if ((Total_Fronts(k,InputNum+1,j).ne.0).and.(Total_Fronts(k+1,InputNum+1,j)==0)) then
	     crowding_dis(k,j)=10000
	  else if ((k.ne.1).and.(Total_Fronts(k,InputNum+1,j).ne.0).and.(Total_Fronts(k+1,InputNum+1,j).ne.0)) &
	  then
	     crowding_dis(k,j)=(((Total_Fronts(k+1,InputNum+1,j)-Total_Fronts(k-1,InputNum+1,j))**2)+&
		 ((Total_Fronts(k+1,InputNum+2,j)-Total_Fronts(k-1,InputNum+2,j))**2))**0.5
	  end if
   end do
   !write(*,*)crowding_dis(:,j)
end do

!Final sorting base on crowding distance=======================================================================================
   DO j=1,SWARM_VariableSelection
      DO k=1,SWARM_VariableSelection
         DO l=k+1,SWARM_VariableSelection
            if (crowding_dis(l,j)>crowding_dis(k,j)) then

	              Total_Frontsj(:)=Total_Fronts(k,:,j)
		          Total_Fronts(k,:,j)=Total_Fronts(l,:,j)
		          Total_Fronts(l,:,j)=Total_Frontsj(:)

	        END if
         END DO
      END DO
   END DO

   l=0
   DO j=1,SWARM_VariableSelection
      DO k=1,SWARM_VariableSelection
         if (Total_Fronts(k,InputNum+1,j).ne.0) then
		    l=l+1
			SelectionMatrix(l,:)=Total_Fronts(k,1:InputNum,j)
			SelectionMatrix_Output_train(l,:)=Total_Fronts(k,InputNum+1:InputNum+NumOfObjFunc,j)
			SelectionMatrix_Output_Valid(l,:)=Total_Fronts(k,InputNum+NumOfObjFunc+1:InputNum+2*NumOfObjFunc,j)
		 end if
      END DO
   END DO

!selection the swarm best and global best=============================================================================

   SelectionMatrix_p(:)=SelectionMatrix(1,:)
   SelectionMatrix_Output_train_p(:)=SelectionMatrix_Output_train(1,:)
   SelectionMatrix_Output_Valid_p(:)=SelectionMatrix_Output_Valid(1,:)
   SelectionMatrix_Output_test_p(:)=SelectionMatrix_Output_test(1,:)

   if (i==1) then
   
      SelectionMatrix_g(:)=SelectionMatrix(1,:)
      SelectionMatrix_Output_train_g(:)=SelectionMatrix_Output_train(1,:)
      SelectionMatrix_Output_valid_g(:)=SelectionMatrix_Output_valid(1,:)
	  SelectionMatrix_Output_test_g(:)=SelectionMatrix_Output_test(1,:)

   else 
       if ((SelectionMatrix_Output_train_g(1)<SelectionMatrix_Output_train_p(1)).and.&
	      (SelectionMatrix_Output_train_g(2)>SelectionMatrix_Output_train_p(2))) then

		  SelectionMatrix_g(:)=SelectionMatrix_p(:)
		  SelectionMatrix_Output_train_g(:)=SelectionMatrix_Output_train_p(:)
		  SelectionMatrix_Output_valid_g(:)=SelectionMatrix_Output_valid_p(:)
		  SelectionMatrix_Output_test_g(:)=SelectionMatrix_Output_test_p(:)

	   end if
   end if
   
   write(100,*)i
   write(100,*)SelectionMatrix_p(:),SelectionMatrix_Output_train_p(:),SelectionMatrix_Output_valid_p(:),SelectionMatrix_Output_&
               test_p(:)
   write(100,*)SelectionMatrix_g(:),SelectionMatrix_Output_train_g(:),SelectionMatrix_Output_valid_g(:),SelectionMatrix_Output_&
               test_g(:)
   write(*,*)i
   write(*,*)SelectionMatrix_p(:),SelectionMatrix_Output_train_p(:),SelectionMatrix_Output_valid_p(:),SelectionMatrix_Output_&
             test_p(:)
   write(*,*)SelectionMatrix_g(:),SelectionMatrix_Output_train_g(:),SelectionMatrix_Output_valid_g(:),SelectionMatrix_Output_&
             test_g(:)

!Velocity Update for input selection:  --------------------------------------------------------------------
!Update selection matrix
      do l=1,SWARM_VariableSelection
         do m=1,InputNum
            CALL random_number(fi_P)
	        CALL random_number(fi_g)
                 V_SelectionMatrix(l,m)=W(l)*V_SelectionMatrix(l,m)+fi_P*Cp_Selection(l)*(SelectionMatrix_P(l)-SelectionMatrix(l,m))&
				 +fi_g*Cg_Selection(l)*(SelectionMatrix_g(m)-SelectionMatrix(l,m)) !the formula of velocity opdate
	             if (abs(b3-a3)<V_SelectionMatrix(l,m)) then
	                V_SelectionMatrix(l,m)=abs(b3-a3) !the limits of velocity
	             elseif (-abs(b3-a3)>V_SelectionMatrix(l,m)) then
	                V_SelectionMatrix(l,m)=-abs(b3-a3) !the limits of velocity
	             end if
				 Sv(l,m)=1/(1+exp(-V_SelectionMatrix(l,m)))
	             
				 CALL random_number(rand3)
				 if (Sv(l,m)>rand3) then
				    SelectionMatrix(l,m)=1 !weights place update
				 else
				    SelectionMatrix(l,m)=0
				 end if
         End do !m
      End do !l

!Mutation==============================================================================================
Mu_No=int(SWARM_VariableSelection*Mutation_Rate)
do j=1,Mu_No
   call random_number(rand3)
   Mu_selected=int(rand3*SWARM_VariableSelection)
   do k=1,InputNum
      call random_number(rand3)
	  if (rand3>0.5) then
	     if (SelectionMatrix(j,k)==0) then
	        SelectionMatrix(j,k)=1
		 else
		    SelectionMatrix(j,k)=0
		 end if
	  end if
   end do
end do


End do !i



!!SUBROUTINES********SUBROUTINES********SUBROUTINES*********SUBROUTINES*********SUBROUTINES
!!SUBROUTINES********SUBROUTINES********SUBROUTINES*********SUBROUTINES*********SUBROUTINES
!!SUBROUTINES********SUBROUTINES********SUBROUTINES*********SUBROUTINES*********SUBROUTINES

!!Neural Network Structure=============================================================================================
!structure of nural network is written by this subroutine. 
contains
Subroutine Structure_output
NeronOutput1(:)=0 !nerons of first layer
NeronOutput2=0 !nerons ofsecond layer
DO ii=1,2
 if (ii==1) then
    DO jj=1,Input_Cunt(j)
	   ll=0
	   DO kk=1,InputNum
	      if (SelectionMatrix(j,kk)==1) then
		     ll=ll+1
	         NeronOutput1(jj)=NeronOutput1(jj)+Xtr(kk,m)*IW(l,(jj-1)*Input_Cunt(j)+ll) !nerons of layer 1 output
		  end if
	   End do !kk
	      NeronOutput1(jj)=NeronOutput1(jj)+b(l,jj) !nerons of layer 1 output
		  NeronOutput1(jj)=2/(1+exp(-2*NeronOutput1(jj)))-1 !nerons of layer 1 output
	End do
 Else if (ii==2) then
    DO kk=1,Input_Cunt(j)
       NeronOutput2=NeronOutput2+NeronOutput1(kk)*LW(l,kk)
	End do !kk
	   NeronOutput2=NeronOutput2+b(l,1+Input_Cunt(j))
 End if
End do !ii
!write(*,*)NeronOutput2
return
End Subroutine Structure_output

!!Neural Network Validation Structure================================================================
Subroutine Valid_output 
NeronOutput1(:)=0
NeronOutput2=0
DO ii=1,2
 if (ii==1) then
    DO jj=1,Input_Cunt(j)
	   ll=0
	   DO kk=1,InputNum
	      if (SelectionMatrix(j,kk)==1) then
		     ll=ll+1
	         NeronOutput1(jj)=NeronOutput1(jj)+Xval(kk,m)*IW(l,(jj-1)*Input_Cunt(j)+ll)
		  end if
	   End do
	      NeronOutput1(jj)=NeronOutput1(jj)+b(l,jj)
		  NeronOutput1(jj)=2/(1+exp(-2*NeronOutput1(jj)))-1
	End do
 Else if (ii==2) then
    DO kk=1,Input_Cunt(j)
       NeronOutput2=NeronOutput2+NeronOutput1(kk)*LW(l,kk)
	End do
	   NeronOutput2=NeronOutput2+b(l,1+Input_Cunt(j))
 End if
End do
return
End Subroutine Valid_output

!!Neural Network Test Structure================================================================
Subroutine Test_output
NeronOutput1(:)=0
NeronOutput2=0
DO ii=1,2
 if (ii==1) then
    DO jj=1,Input_Cunt(j)
	   ll=0
	   DO kk=1,InputNum
	      if (SelectionMatrix(j,kk)==1) then
		     ll=ll+1
	         NeronOutput1(jj)=NeronOutput1(jj)+Xts(kk,m)*IW(l,(jj-1)*Input_Cunt(j)+ll)
		  end if
	   End do
	      NeronOutput1(jj)=NeronOutput1(jj)+b(l,jj)
		  NeronOutput1(jj)=2/(1+exp(-2*NeronOutput1(jj)))-1
	End do
 Else if (ii==2) then
    DO kk=1,Input_Cunt(j)
       NeronOutput2=NeronOutput2+NeronOutput1(kk)*LW(l,kk)
	End do
	   NeronOutput2=NeronOutput2+b(l,1+Input_Cunt(j))
 End if
End do
return
End Subroutine Test_output

!!ERROR CALCULATION ==================================================================================================
Subroutine ERROR_Calculator
ERROR=0
DO ii=1,trainNumber
   ERROR(l)=ERROR(l)+((Ytr_Pre(ii)-Ytr(ii))**2)
End do
MSE(l)=ERROR(l)/TrainNumber
NSE(l)=1-ERROR(l)/sum1
return
End Subroutine ERROR_Calculator

!!Validation ERROR CALCULATION =============================================================
Subroutine Valid_ERROR_Calculator
ERROR=0
DO ii=1,ValidNumber
   ERROR(l)=ERROR(l)+((Yval_Pre(ii)-Yval(ii))**2)
End do
MSE(l)=ERROR(l)/ValidNumber
NSE(l)=1-ERROR(l)/sum7
return
End Subroutine Valid_ERROR_Calculator


!!Test ERROR CALCULATION =============================================================
Subroutine Test_ERROR_Calculator
ERROR=0
DO ii=1,TestNumber
   ERROR(l)=ERROR(l)+((Yts_Pre(ii)-Yts(ii))**2)
End do
MSE(l)=ERROR(l)/TestNumber
NSE(l)=1-ERROR(l)/sum2
return
End Subroutine Test_ERROR_Calculator

!!SORT POPULATION =============================================================
!sorting population from the best output to the worst
Subroutine SORTing
DO ii=1,SWARM_NN_train
  DO jj=ii,SWARM_NN_train
     if (MSE(jj)<MSE(ii)) then

	    MSEj=MSE(ii)
		MSE(ii)=MSE(jj)
		MSE(jj)=MSEj
		
	    NSEj=NSE(ii)
		NSE(ii)=NSE(jj)
		NSE(jj)=NSEj
		
		!Wj=W(ii)
		!W(ii)=W(jj)
		!W(jj)=Wj

		!Cpj=Cp(ii)
		!Cp(ii)=Cp(jj)
		!Cp(jj)=Cpj

		!Cgj=Cg(ii)
		!Cg(ii)=Cg(jj)
		!Cg(jj)=Cgj

		IWj=IW(ii,:)
		IW(ii,:)=IW(jj,:)
		IW(jj,:)=IWj(:)

		lWj=lW(ii,:)
		lW(ii,:)=lW(jj,:)
		lW(jj,:)=lWj(:)

		bj=b(ii,:)
		b(ii,:)=b(jj,:)
		b(jj,:)=bj(:)
	 END if
  END DO
END DO

return
End Subroutine SORTing

!!SORT POPULATION by multi objective Paretto Front =============================================================
!sorting population from the best output to the worst
Subroutine Paretto_SORTing
Nq(:)=0
DO ii=1,SWARM_VariableSelection
   if (ii==1) then
      Do jj=2,SWARM_VariableSelection
	     if (((SelectionMatrix_Output_train(ii,1))>=(SelectionMatrix_Output_train(jj,1)))) then
         !if (((SelectionMatrix_Output_test(ii,1))>=(SelectionMatrix_Output_test(jj,1))).and.((SelectionMatrix_Output_test&
			!(ii,2))<=(SelectionMatrix_Output_test(jj,2)))) then
		          
		          Nq(jj)=Nq(jj)+1		  

         end if
      end do !jj
   else if (ii==SWARM_VariableSelection) then
           Do jj=1,SWARM_VariableSelection-1
		      if (((SelectionMatrix_Output_train(ii,1))>=(SelectionMatrix_Output_train(jj,1)))) then
              !if (((SelectionMatrix_Output_test(ii,1))>=(SelectionMatrix_Output_test(jj,1))).and.((SelectionMatrix_Output_test&
			     !(ii,2))<=(SelectionMatrix_Output_test(jj,2)))) then
		            
		          Nq(jj)=Nq(jj)+1
		   
              end if
           end do !jj
   else
       Do jj=1,ii-1
	      if (((SelectionMatrix_Output_train(ii,1))>=(SelectionMatrix_Output_train(jj,1)))) then
          !if (((SelectionMatrix_Output_test(ii,1))>=(SelectionMatrix_Output_test(jj,1))).and.((SelectionMatrix_Output_test&
			 !(ii,2))<=(SelectionMatrix_Output_test(jj,2)))) then
		       
		          Nq(jj)=Nq(jj)+1

          end if
       end do !jj
	   Do jj=ii+1,SWARM_VariableSelection
	      if (((SelectionMatrix_Output_train(ii,1))>=(SelectionMatrix_Output_train(jj,1)))) then
          !if (((SelectionMatrix_Output_test(ii,1))>=(SelectionMatrix_Output_test(jj,1))).and.((SelectionMatrix_Output_test&
			 !(ii,2))<=(SelectionMatrix_Output_test(jj,2)))) then
		       
		          Nq(jj)=Nq(jj)+1

          end if
       end do !jj
   end if
End do !ii
return
End Subroutine Paretto_SORTing


end program Mo_PSO_IVS

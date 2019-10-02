subroutine nrmaxl(rfin,maxk,rinit,kopt,vi)
  implicit none
  integer,intent(in) :: maxk
  real*8,intent(in)  :: rinit
  integer,intent(in) :: kopt
  real*8,intent(in)  :: vi(maxk)
  real*8,intent(out)  :: rfin
  real*8 :: b,L0,a,stepmax,ga,gb,Cov2(2,2),Covinv2(2,2)
  real*8 :: jf,t,s,tt,func,sigma,sa,sb
  integer :: j,niter
  !  write (6,*) maxk,rinit,kopt,(vi(j),j=1,kopt)
  b=rinit
  L0=0.
  a=0.
  stepmax=0.1*abs(b)
  gb=float(kopt)
  ga=float(kopt+1)*float(kopt)/2.
  Cov2(1,1)=0. !gbb
  Cov2(1,2)=0. !gab
  Cov2(2,2)=0. !gaa
  do j=1,kopt
    jf=float(j)
    t=b+a*jf
    s=exp(t)
    tt=vi(j)*s
    L0=L0+t-tt
    gb=gb-tt
    ga=ga-jf*tt
    Cov2(1,1)=Cov2(1,1)-tt
    Cov2(1,2)=Cov2(1,2)-jf*tt
    Cov2(2,2)=Cov2(2,2)-jf*jf*tt
  enddo
  Cov2(2,1)=Cov2(1,2)
  Covinv2=matinv2(Cov2)
  func=100.
  niter=0
  do while ((func>1D-3).and.(niter.lt.1000))
    sb=(Covinv2(1,1)*gb+Covinv2(1,2)*ga)
    sa=(Covinv2(2,1)*gb+Covinv2(2,2)*ga)
    niter=niter+1
    sigma=0.1
    if (abs(sigma*sb).gt.stepmax) then
      sigma=abs(stepmax/sb)
    endif
    b=b-sigma*sb
    a=a-sigma*sa
    L0=0.
    gb=float(kopt)
    ga=float(kopt+1)*float(kopt)/2.
    Cov2(1,1)=0. !gbb
    Cov2(1,2)=0. !gab
    Cov2(2,2)=0. !gaa
    do j=1,kopt
      jf=float(j)
      t=b+a*jf
      s=exp(t)
      tt=vi(j)*s
      L0=L0+t-tt
      gb=gb-tt
      ga=ga-jf*tt
      Cov2(1,1)=Cov2(1,1)-tt
      Cov2(1,2)=Cov2(1,2)-jf*tt
      Cov2(2,2)=Cov2(2,2)-jf*jf*tt
    enddo
    Cov2(2,1)=Cov2(1,2)
    Covinv2=matinv2(Cov2)
    if ((abs(a).le.tiny(a)).or.(abs(b).le.tiny(b))) then
      func=max(gb,ga)
    else
      func=max(abs(gb/b),abs(ga/a))
    endif
  enddo
  Cov2(:,:)=-Cov2(:,:)
  Covinv2=matinv2(Cov2)
  rfin=b
contains
  pure function matinv2(A) result(B)
!! Performs a direct calculation of the inverse of a 2×2 matrix.
    real*8, intent(in) :: A(2,2)   !! Matrix
    real*8             :: B(2,2)   !! Inverse matrix
    real*8             :: detinv
    detinv = 1./(A(1,1)*A(2,2) - A(1,2)*A(2,1))
    B(1,1) = +detinv * A(2,2)
    B(2,1) = -detinv * A(2,1)
    B(1,2) = -detinv * A(1,2)
    B(2,2) = +detinv * A(1,1)
  end function
end subroutine NRmaxL


#include<stdio.h>
#include<stdlib.h>
typedef int EmelType;
//结点结构体
typedef struct LNode{
    EmelType data;
    struct LNode *next;
}LNode,*LinkList;

void HeadInsert(LinkList &H){
    EmelType x;
    H=(LinkList)malloc(sizeof(LNode));//申请头结点空间
    LinkList p;
    p=(LinkList)malloc(sizeof(LNode));
    p->next=NULL;
    scanf("%d",&x);
    while(x!=9999){
        p->data=x;
        H->next=p;
        p=(LinkList)malloc(sizeof(LNode));
        p->next=H->next;
        scanf("%d",&x);
    }
}
void PrintList(LinkList H){
    H=H->next;
    //若该空间地址不为NULL则输出
    while(H!=NULL){
        printf("%3d",H->data);
        H=H->next;
    }
}
//头插法新建链表
int main(){
    LinkList H;//头指针
    HeadInsert(H);
    PrintList(H);
    return 0;
}
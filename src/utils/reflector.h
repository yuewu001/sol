/*************************************************************************
	> File Name: reflector.h
	> Copyright (C) 2014 Yue Wu<yuewu@outlook.com>
	> Created Time: 2014/5/12 Monday 16:21:00
	> Functions: C++ reflector
 ************************************************************************/
#ifndef HEADER_CPP_REFLECTOR
#define HEADER_CPP_REFLECTOR

#include <map>
#include <string>

namespace BOC{

	//declaration of ClassInfo
	class ClassInfo;
	//declaration
	class Registry;

	static std::map<std::string, ClassInfo*> mapClassInfo;

	//declaration of function to create a new instance
	typedef void* (*CreateFunction)(void* lossFunc);

//	//declaration of register function
	inline bool Register(ClassInfo* classInfo);

	class ClassInfo {
	protected:
		std::string type;
		std::string description;
		CreateFunction func;

	public:
		const std::string GetType() const { return this->type; }
		const std::string GetDescr() const { return this->description; }
	public:
		ClassInfo(std::string type, std::string descr,  CreateFunction func) :
			type(type), description(descr), func(func){
			Register(this);
		}

		void* CreateObject(void* lossFunc) const {
			return func ? (*func)(lossFunc) : NULL;
		}
	};

	class Registry{
	public:
		static bool Register(ClassInfo* classInfo){
		if (classInfo != NULL){
			if (mapClassInfo.find(classInfo->GetType()) == mapClassInfo.end()){
				mapClassInfo[classInfo->GetType()] = classInfo;
				return true;
			}
		}
		return false;
	}
		static void* CreateObject(std::string name, void *lossFunc){
			std::map<std::string, ClassInfo* >::iterator iter = mapClassInfo.find(name);
			if (iter != mapClassInfo.end()){
				return ((ClassInfo*)(iter->second))->CreateObject(lossFunc);
			}
			return NULL;
		}
	};
	inline bool Register(ClassInfo* classInfo) {
		return Registry::Register(classInfo);
	}

#define DECLARE_CLASS \
public: \
	static ClassInfo classInfo; \
public:\
	static void* CreateObject(void *param1); \
	static ClassInfo& GetClassMsg() { return classInfo; }

#define IMPLEMENT_CLASS(name) \
	template <typename FeatType, typename LabelType> \
	ClassInfo name<FeatType, LabelType>::classInfo(#name, "", name<FeatType, LabelType>::CreateObject); \
	\
	template <typename FeatType, typename LabelType> \
	void* name<FeatType, LabelType>::CreateObject(void *lossFunc) \
	{ return new name<FeatType, LabelType>((LossFunction<FeatType, LabelType>*)lossFunc); }

#define APPEND_INFO(info,name,T1,T2) \
	info.append("\n\t"); \
	info.append(name<T1,T2>::Id_Str());
	//info.append(name<T1,T2>::GetClassMsg().GetDescr());
}

#endif

from copy import deepcopy

def default_cond(section,state):
    return True

def default_get_func_args(section,state):
    return state,None

def default_func_apply(jbobj,path,func_args):
    return jbobj

def get_path_b2(section,state={},cond=None,get_func_args=lambda x,y:(y,None)):
    # try:
    cond = default_cond if cond is None else cond
    get_func_args = default_get_func_args if get_func_args is None else get_func_args
    
    def clean_path(p):
        return '.'.join([i for i in p.split('.') if i!=''])

    def update_state(state,name,p_name):
        n_state=deepcopy(state)
        n_state['name']=name
        n_state['p_name']=p_name
        n_state['a_p_path']=clean_path(f'{n_state["a_p_path"]}.{p_name}')
        n_state['p_path']=clean_path(f'{n_state["p_path"]}.{p_name}')
        n_state['path']=clean_path(f'{n_state["p_path"]}.{name}')
        n_state['a_path']=clean_path(f'{n_state["a_p_path"]}.{name}')
        return n_state
    all_paths={}
    prop_paths={}
    arr_paths={}
    if not state:
        state = {k:'' for k in ['sname','name','p_name','p_path','a_p_path','path','a_path']}
    name=state['name']
    title=section.get('title',name)
    title = title if name == '' else name
    stype=section.get('type','')
    if cond(section,state):
        state,func_args=get_func_args(section,state)
        prop_paths.update({title:[state,func_args,'property']})
    if stype=='object':
        for prop_name,prop_section in section.get('properties',{}).items():
            prop_state=update_state(state,prop_name,name)
            prop_state['p_path']=clean_path(f'{prop_state["p_path"]}.properties')
            prop_paths.update(get_path_b2(prop_section,prop_state,cond,get_func_args))
        for i, v in prop_paths.items(): v[0]['sname'] = clean_path(f'{title}.{i}')
        # ppaths={f'{title}.{i}':v for i,v in ppaths.items()}
        prop_paths={v[0]['a_path']:v for i,v in prop_paths.items()}
        # print(ppaths)
        for idx,isection in enumerate(section.get('allOf',[])):
            # all_paths.update(get_path_b2(isection,'',cond,get_func_args))
            all_state=update_state(state,'',name)
            all_state['p_path']=clean_path(f'{all_state["p_path"]}.allOf[{idx}]')
            # ipath=get_path_b2(isection,all_state,cond,get_func_args)
            all_paths.update(get_path_b2(isection,all_state,cond,get_func_args))
        for i, v in all_paths.items(): v[0]['sname'] = clean_path(f'{title}.{".".join(i.split(".")[:])}')
        all_paths.update({v[0]['a_path']:v for i,v in all_paths.items()})
        # print('all_paths',all_paths)
        # all_paths={f'{title}.allOf[n].{".".join(i.split(".")[:])}':[f'{jtitle}.{v[0]}',v[1],v[2]] for i,v in all_paths.items()}
        # print('all_paths',all_paths)
    elif stype=='array':
        arr_state=update_state(state,'',name)
        arr_state['a_p_path']=clean_path(f'{arr_state["a_p_path"]}[n]')
        arr_paths=get_path_b2(section['items'],arr_state,cond,get_func_args)
        # print('paths:',paths)
        for i, v in arr_paths.items(): v[0]['sname'] = clean_path(f'{title}.{i}')
        # paths={f'{title}.{i}':[*v[0:-1],'array'] for i,v in paths.items()}
        arr_paths={v[0]['a_path']:[*v[0:-1],'array'] for i,v in arr_paths.items()}
        
        # print('paths:',paths)
   
    # elif cond(section):
    #     return {title:[jtitle,get_func_args(section),'property']}
    # else:
    #     return {}
    all_paths.update(prop_paths)
    # ppaths.update()
    all_paths.update(arr_paths)
    # print(ppaths)
    # for k,v in ppaths.items():
    #     print(v[0],'.'.join([i for i in v[0].split('.') if i!='']))
    return all_paths

def update_archive_b2(jbobj,paths,func_apply=None):
    func_apply = default_func_apply if func_apply is None else func_apply
    def check_path(path):
        return path in jbobj
    if jbobj is None:
        return None
    if isinstance(jbobj,list):
        return [update_archive_b2(i,paths,func_apply) for i in jbobj]
    for path,(state,func_args,stype) in paths.items():
        if stype=='property' and check_path(path):
            # print(path,paths[path])
            jbobj=func_apply(jbobj,path,func_args)
        elif stype=='array':
            # print(path,paths[path])
            sub_paths=path.split('[n]',maxsplit=1)
            if check_path(sub_paths[0]):
                # print(sub_paths,paths[path])
                if len(sub_paths)>1:
                    n_stype = 'array' if '[n]' in sub_paths[1][1:] else 'property'
                    n_state=deepcopy(state)
                    n_state['a_path']=sub_paths[1][1:]
                    n_paths={n_state['a_path']:[n_state,func_args,n_stype]}
                    jbobj[sub_paths[0]]=update_archive_b2(jbobj[sub_paths[0]],n_paths,func_apply)
                else:
                    jbobj=func_apply(jbobj,sub_paths[0],func_args)
    return jbobj
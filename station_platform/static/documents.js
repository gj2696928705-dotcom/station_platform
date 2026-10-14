"use strict";
let documentState=null;

// Escape every text fragment. Raw HTML, images and executable URL schemes are
// never interpreted. This intentionally supports a small offline Markdown set.
function markdownInline(text){
  const pattern=/`([^`\n]+)`|\*\*([^*\n]+)\*\*|\[([^\]\n]+)\]\(([^)\s]+)\)/g;
  let out="",end=0;
  for(const match of text.matchAll(pattern)){
    out+=E(text.slice(end,match.index));
    if(match[1]!==undefined)out+=`<code>${E(match[1])}</code>`;
    else if(match[2]!==undefined)out+=`<strong>${E(match[2])}</strong>`;
    else {
      let safe=false;
      try{safe=["https:","http:"].includes(new URL(match[4]).protocol);}catch{}
      out+=safe?`<a href="${E(match[4])}" target="_blank" rel="noopener noreferrer">${E(match[3])}</a>`:E(match[0]);
    }
    end=match.index+match[0].length;
  }
  return out+E(text.slice(end));
}
function renderMarkdown(body){
  const lines=String(body).replace(/\r\n?/g,"\n").split("\n");
  const out=[];let paragraphs=[],list=[],listType=null,code=null;
  const flushParagraph=()=>{if(paragraphs.length){out.push(`<p>${paragraphs.map(markdownInline).join("<br>")}</p>`);paragraphs=[];}};
  const flushList=()=>{if(list.length){out.push(`<${listType}>${list.map(item=>`<li>${markdownInline(item)}</li>`).join("")}</${listType}>`);list=[];listType=null;}};
  for(const line of lines){
    if(/^\s*```/.test(line)){
      flushParagraph();flushList();
      if(code!==null){out.push(`<pre><code>${E(code.join("\n"))}</code></pre>`);code=null;}else code=[];
      continue;
    }
    if(code!==null){code.push(line);continue;}
    const heading=line.match(/^(#{1,6})\s+(.+)$/),bullet=line.match(/^\s*[-*+]\s+(.+)$/),ordered=line.match(/^\s*\d+\.\s+(.+)$/);
    if(!line.trim()){flushParagraph();flushList();continue;}
    if(heading){flushParagraph();flushList();out.push(`<h${heading[1].length}>${markdownInline(heading[2])}</h${heading[1].length}>`);continue;}
    if(bullet||ordered){flushParagraph();const type=bullet?"ul":"ol";if(listType&&listType!==type)flushList();listType=type;list.push((bullet||ordered)[1]);continue;}
    flushList();
    if(/^>\s?/.test(line)){flushParagraph();out.push(`<blockquote>${markdownInline(line.replace(/^>\s?/,""))}</blockquote>`);}
    else paragraphs.push(line);
  }
  if(code!==null)out.push(`<pre><code>${E(code.join("\n"))}</code></pre>`);
  flushParagraph();flushList();return out.join("");
}

async function agentEditor(id){
  if(!state.tools.length)state.tools=await api("/api/tools");
  const a=state.data.agents.find(a=>a.id===id)||{name:"",prompt:"你是场站平台的专业助手。先查询工具事实，再给出有依据的解释。",allowed_tools:["station_state"],model_connection:null,max_turns:12,enabled:true};
  const role=a.builtin?a.id:"custom";
  document.querySelector("#editor-content").innerHTML=`<form id="agent-form"><div class="panel-head"><h2>${id?"编辑 Agent 配置":"新增 Agent"}</h2><button class="btn small" type="button" data-action="editor-close">关闭</button></div><div class="form-grid"><div class="field"><label for="agent-name">名称</label><input id="agent-name" name="name" value="${E(a.name)}" required></div><div class="field"><label for="agent-model">模型连接</label><select id="agent-model" name="model_connection"><option value="">使用平台默认连接</option>${state.data.models.map(m=>`<option value="${E(m.id)}" ${a.model_connection===m.id?"selected":""}>${E(m.name)}</option>`).join("")}</select></div>
  ${id?'<div class="notice-box field full">运行职责请在“查看文档”中编辑、保存草稿并发布。本页仅管理角色配置。</div>':`<div class="field full"><label for="agent-initial-body">初始运行职责</label><textarea id="agent-initial-body" name="prompt" rows="8" maxlength="24000" required>${E(a.prompt)}</textarea><small class="muted">创建时登记为首版文档；后续职责修改统一在文档页发布。</small></div>`}
  <div class="field"><label for="agent-turns">最大工具交互轮数</label><input id="agent-turns" name="max_turns" type="number" min="1" max="30" value="${a.max_turns}"></div><div class="field"><label for="agent-enabled">启用状态</label><select id="agent-enabled" name="enabled"><option value="true" ${a.enabled?"selected":""}>启用</option><option value="false" ${!a.enabled?"selected":""}>停用</option></select></div><div class="field full"><label>可调用工具</label><div class="checks">${state.tools.filter(t=>t.roles.includes(role)).map(t=>`<label class="check"><input type="checkbox" name="tool" value="${E(t.name)}" ${a.allowed_tools.includes(t.name)?"checked":""}><span>${E(t.name)}<small>${E(t.description)}</small></span></label>`).join("")}</div></div></div><div class="form-foot"><button class="btn primary" type="submit">保存配置</button><span class="muted">计划发布权限固定属于 O1。</span></div></form>`;
  document.querySelector("#editor").showModal();
  document.querySelector("#agent-form").onsubmit=async e=>{
    e.preventDefault();const form=e.target,f=new FormData(form),btn=form.querySelector("button[type=submit]");
    const data={name:f.get("name"),model_connection:f.get("model_connection")||null,max_turns:Number(f.get("max_turns")),enabled:f.get("enabled")==="true",allowed_tools:f.getAll("tool")};
    if(id)data.expected_version=a.version;else data.prompt=f.get("prompt");
    btn.disabled=true;
    try{await api(id?"/api/agents/"+id:"/api/agents",id?"PUT":"POST",data);document.querySelector("#editor").close();await refresh();renderAgents();toast("角色配置已保存，后续新会话和任务使用新版本。");}
    catch(error){toast(error.message);}finally{btn.disabled=false;}
  };
}

function setDocumentData(data,options={}){
  const previous=documentState;
  documentState={data,tab:options.tab||previous?.tab||"current",busy:false,message:options.message||"",
    editBody:options.keepBody?previous.editBody:(data.draft?.body??data.body),
    baseAgentVersion:options.rebase?data.agent_version:(data.draft?.base_agent_version??data.agent_version),
    baseDocumentRevision:options.rebase?data.revision:(data.draft?.base_document_revision??data.revision)};
}
async function openDocuments(id){
  const result=await api(`/api/agents/${id}/documents`);
  documentState=null;setDocumentData(result.documents[0],{tab:"current"});renderDocumentEditor();
  document.querySelector("#document-editor").showModal();
}
function documentVersions(){
  return {base_agent_version:documentState.baseAgentVersion,base_document_revision:documentState.baseDocumentRevision,expected_draft_revision:documentState.data.draft_revision};
}
function documentDirty(){const s=documentState;return s.editBody!==(s.data.draft?.body??s.data.body);}
function documentPublishable(){const s=documentState;return Boolean(s.data.draft&&!documentDirty()&&!s.data.draft.stale&&s.baseAgentVersion===s.data.agent_version);}
function updateDocumentButtons(){
  const s=documentState,save=document.querySelector("#document-save"),publish=document.querySelector("#document-publish"),status=document.querySelector("#document-edit-status");
  if(save)save.disabled=s.busy||!s.editBody.trim();
  if(publish)publish.disabled=s.busy||!documentPublishable();
  if(status)status.textContent=documentDirty()?"编辑内容尚未保存。发布前请先保存草稿。":s.data.draft?(s.data.draft.stale?"草稿基于旧配置；刷新版本信息并比较后重新保存。":"草稿已保存，尚未用于运行。") :"当前编辑内容与生效正文一致；修改后先保存草稿。";
}
function renderDocumentEditor(){
  const s=documentState,d=s.data;
  let content="";
  if(s.tab==="current")content=`<div class="doc-meta"><div><span>原始来源</span><strong>${E(d.source.reference)}</strong><small>${E(d.source.description)}</small></div><div><span>当前存放位置</span><strong>${E(d.storage)}</strong><small>这是实际使用的正文，没有对应的独立磁盘文件。</small></div><div><span>当前版本</span><strong>文档 r${d.revision} · 角色 v${d.agent_version}</strong><small>正文登记或发布于 ${dt(d.published_at)}，对应角色 v${d.published_agent_version}</small></div></div><section class="doc-section"><div class="panel-head"><h3>当前生效正文</h3>${badge("用于运行","green")}</div><div class="markdown doc-published" data-testid="published-document">${renderMarkdown(d.body)}</div><details class="doc-raw"><summary>查看原始正文（逐字内容）</summary><pre>${E(d.body)}</pre></details></section><div class="doc-injection"><strong>平台追加信息 · 只读</strong><code>${E(d.runtime_injection.text)}</code><small>${E(d.runtime_injection.source)}</small></div>`;
  if(s.tab==="draft")content=`<div class="notice-box">保存草稿不影响运行。发布后的正文用于新会话和新任务；已有会话和任务继续使用原版本。</div><div class="doc-edit-grid"><div class="field"><label for="document-body">运行职责草稿</label><textarea id="document-body" maxlength="24000" spellcheck="false">${E(s.editBody)}</textarea><small id="document-edit-status" class="muted"></small></div><section class="doc-preview"><h3>Markdown 预览</h3><div id="document-preview" class="markdown">${renderMarkdown(s.editBody)}</div></section></div><details class="doc-raw"><summary>比较当前生效正文与服务器已保存草稿</summary><div class="doc-comparison"><div><strong>当前生效 · r${d.revision}</strong><pre>${E(d.body)}</pre></div><div><strong>服务器草稿 · ${d.draft?"修订 "+d.draft.revision:"未保存"}</strong><pre>${E(d.draft?.body??"尚无草稿")}</pre></div></div></details><div class="doc-actions"><button class="btn" id="document-save" data-action="document-save">保存草稿</button><button class="btn primary" id="document-publish" data-action="document-publish">发布为新版本</button><button class="btn" data-action="document-refresh">刷新版本信息</button><span class="muted">编辑基于角色 v${s.baseAgentVersion}、文档 r${s.baseDocumentRevision}</span></div>`;
  if(s.tab==="history")content=`<p class="muted">历史记录只读。首次登记保留当时的实际正文；配置变化不改写已发布的历史正文。</p><div class="doc-history">${d.history.map(h=>`<details><summary><strong>文档 r${h.revision} · 角色 v${h.agent_version}</strong><span>${h.reason==="baseline"?"首次登记":h.reason==="created"?"角色创建":"网页发布"} · ${dt(h.published_at)}</span></summary><div class="markdown">${renderMarkdown(h.body)}</div><details class="doc-raw"><summary>原始正文</summary><pre>${E(h.body)}</pre></details></details>`).join("")}</div>`;
  document.querySelector("#document-content").innerHTML=`<div class="doc-header"><div><div class="eyebrow">${E(d.agent_id)} · ${d.scope==="station"?"独立站级节点":"运营商中心"}</div><h2 id="document-heading">${E(d.agent_name)} · ${E(d.title)}</h2><div class="actions">${badge("角色 v"+d.agent_version)}${badge("文档 r"+d.revision)}${d.draft?badge(d.draft.stale?"草稿需要更新":"有未发布草稿","amber"):badge("已发布","green")}</div></div><button class="btn small" data-action="document-close">关闭</button></div><div class="doc-tabs" role="tablist" aria-label="运行文档视图">${[["current","当前生效"],["draft","草稿编辑"],["history","发布历史"]].map(([tab,label])=>`<button role="tab" aria-selected="${s.tab===tab}" class="${s.tab===tab?"active":""}" data-action="document-tab" data-tab="${tab}">${label}</button>`).join("")}</div><div class="doc-body">${s.message?`<div class="doc-message" role="status">${E(s.message)}</div>`:""}${content}</div>`;
  const input=document.querySelector("#document-body");
  if(input)input.oninput=()=>{documentState.editBody=input.value;document.querySelector("#document-preview").innerHTML=renderMarkdown(input.value);updateDocumentButtons();};
  updateDocumentButtons();
}
document.addEventListener("click",async e=>{
  const el=e.target.closest("[data-action^='document-']");if(!el||el.disabled||!documentState)return;
  const action=el.dataset.action,s=documentState,id=s.data.agent_id;
  if(action==="document-close"){document.querySelector("#document-editor").close();return;}
  if(s.busy)return;
  if(action==="document-tab"){s.tab=el.dataset.tab;renderDocumentEditor();return;}
  s.busy=true;el.disabled=true;updateDocumentButtons();
  try{
    if(action==="document-refresh"){
      const result=await api(`/api/agents/${id}/documents`);
      setDocumentData(result.documents[0],{keepBody:true,rebase:true,message:"已刷新版本信息，当前编辑内容保留。请比较当前生效正文与服务器草稿后，再保存。"});
    }
    if(action==="document-save"){
      const result=await api(`/api/agents/${id}/documents/instructions/draft`,"PUT",{...documentVersions(),body:s.editBody});
      setDocumentData(result,{tab:"draft",message:"草稿已保存。当前运行正文保持原版本。"});
    }
    if(action==="document-publish"){
      if(!documentPublishable())throw new Error("请先保存当前编辑内容，再发布。");
      const result=await api(`/api/agents/${id}/documents/instructions/publish`,"POST",documentVersions());
      setDocumentData(result,{tab:"current",message:`已发布文档 r${result.revision}，角色更新为 v${result.agent_version}。之后创建的会话和任务使用新正文。`});
      await refresh();if(state.page==="agents")renderAgents();
    }
  }catch(error){s.message=error.message;documentState=s;}
  finally{documentState.busy=false;renderDocumentEditor();}
});
